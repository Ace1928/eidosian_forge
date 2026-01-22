import collections
import dataclasses
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class NeuronCoreStats:
    """AWS Trainium stats."""
    name: str = 'trn.{key}'
    samples: 'Deque[_Stats]'

    def write_neuron_monitor_config(self) -> None:
        """Write neuron monitor config file."""
        pathlib.Path(self.neuron_monitor_config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.neuron_monitor_config_path, 'w') as f:
            json.dump(NEURON_MONITOR_DEFAULT_CONFIG, f, indent=4)

    def neuron_monitor(self) -> None:
        """Run neuron-monitor in a separate process to collect raw data."""
        self.write_neuron_monitor_config()
        try:
            command = [NEURON_MONITOR_PATH, '-c', self.neuron_monitor_config_path]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None) as process:
                while not self.shutdown_event.is_set():
                    if process.stdout is None:
                        self.shutdown_event.wait(0.1)
                        continue
                    raw_data = process.stdout.readline()
                    if raw_data:
                        self.raw_samples.append(raw_data)
                process.kill()
                process.wait()
        except Exception as e:
            logger.error('neuron-monitor failed: %s' % e)

    def __init__(self, pid: int, neuron_monitor_config_path: Optional[str]) -> None:
        self.pid = pid
        self.neuron_monitor_config_path = neuron_monitor_config_path or tempfile.NamedTemporaryFile(delete=False).name
        self.raw_samples: Deque[bytes] = deque(maxlen=10)
        self.samples: Deque[_Stats] = deque()
        self.shutdown_event = threading.Event()
        self.neuron_monitor_thread: Optional[threading.Thread] = None

    def setup(self) -> None:
        """Start the neuron-monitor thread for collecting raw data."""
        if self.neuron_monitor_thread is not None:
            return
        logger.debug('Starting neuron-monitor thread')
        self.shutdown_event.clear()
        self.neuron_monitor_thread = threading.Thread(name='NeuronCoreMntr', target=self.neuron_monitor, daemon=True)
        self.neuron_monitor_thread.start()

    def teardown(self) -> None:
        """Stop the neuron-monitor thread."""
        logger.debug('Stopping neuron-monitor thread')
        try:
            self.shutdown_event.set()
            assert self.neuron_monitor_thread is not None
            self.neuron_monitor_thread.join()
        except Exception as e:
            logger.error('neuron-monitor thread failed to stop: %s' % e)
        finally:
            self.neuron_monitor_thread = None

    def _is_matching_entry(self, entry: dict) -> bool:
        """Check if the entry should be saved.

        Checks if the pid in the entry matches the pid of the process.
        If not (as in the case of multi-process training with torchrun),
        checks if the LOCAL_RANK environment variable is set.

        todo: add matching by neuron_runtime_tag
        """
        return int(entry['pid']) == int(self.pid) or 'LOCAL_RANK' in os.environ

    def sample(self) -> None:
        try:
            raw_stats = json.loads(self.raw_samples[-1])
            neuron_runtime_data = [entry['report'] for entry in raw_stats['neuron_runtime_data'] if self._is_matching_entry(entry)][0]
            neuroncores_in_use = neuron_runtime_data['neuroncore_counters']['neuroncores_in_use']
            neuroncore_utilization = {int(k): v['neuroncore_utilization'] for k, v in neuroncores_in_use.items()}
            neuron_runtime_used_bytes = neuron_runtime_data['memory_used']['neuron_runtime_used_bytes']
            host_total_memory_usage = neuron_runtime_used_bytes['host']
            neuron_device_total_memory_usage = neuron_runtime_used_bytes['neuron_device']
            usage_breakdown = neuron_runtime_used_bytes['usage_breakdown']
            host_memory_usage = _HostMemoryUsage(**usage_breakdown['host'])
            neuroncore_memory_usage = {int(k): _NeuronCoreMemoryUsage(**v) for k, v in usage_breakdown['neuroncore_memory_usage'].items()}
            local_rank = int(os.environ.get('LOCAL_RANK', -1337))
            if local_rank >= 0:
                neuroncore_utilization = {local_rank: neuroncore_utilization[local_rank]}
                neuroncore_memory_usage = {local_rank: neuroncore_memory_usage[local_rank]}
            stats: _Stats = _Stats(neuroncore_utilization=neuroncore_utilization, host_total_memory_usage=host_total_memory_usage, neuron_device_total_memory_usage=neuron_device_total_memory_usage, host_memory_usage=host_memory_usage, neuroncore_memory_usage=neuroncore_memory_usage)
            self.samples.append(stats)
        except Exception as e:
            pass

    def clear(self) -> None:
        self.samples.clear()

    @staticmethod
    def flatten_stats(sample: _Stats) -> dict:
        """Flatten _Stats object into a flat dict of numbers."""
        flattened = {}

        def helper(key: str, value: Any) -> None:
            if isinstance(value, (int, float)):
                ret = {f'{key}': value}
                flattened.update(ret)
                return
            elif isinstance(value, dict):
                for kk, vv in value.items():
                    if isinstance(kk, int):
                        helper(f'{kk}.{key}', vv)
                    else:
                        helper(f'{key}.{kk}', vv)
                return
            elif isinstance(value, list):
                for i, val in enumerate(value):
                    helper(f'{i}.{key}', val)
        for kkk, vvv in dataclasses.asdict(sample).items():
            helper(kkk, vvv)
        return flattened

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        stats = {}
        merged_samples: Dict[str, List[Union[int, float]]] = collections.defaultdict(list)
        for flattened_sample in (self.flatten_stats(sample) for sample in self.samples):
            for k, v in flattened_sample.items():
                merged_samples[k].append(v)
        for k, v in merged_samples.items():
            stats[self.name.format(key=k)] = aggregate_mean(v)
        return stats