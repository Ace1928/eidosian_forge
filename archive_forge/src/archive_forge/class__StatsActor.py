import collections
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import numpy as np
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.util import capfirst
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@ray.remote(num_cpus=0)
class _StatsActor:
    """Actor holding stats for blocks created by LazyBlockList.

    This actor is shared across all datasets created in the same cluster.
    In order to cap memory usage, we set a max number of stats to keep
    in the actor. When this limit is exceeded, the stats will be garbage
    collected in FIFO order.

    TODO(ekl) we should consider refactoring LazyBlockList so stats can be
    extracted without using an out-of-band actor."""

    def __init__(self, max_stats=1000):
        self.metadata = collections.defaultdict(dict)
        self.last_time = {}
        self.start_time = {}
        self.max_stats = max_stats
        self.fifo_queue = []
        self.next_dataset_id = 0
        self.datasets: Dict[str, Any] = {}
        op_tags_keys = ('dataset', 'operator')
        self.bytes_spilled = Gauge('data_spilled_bytes', description='Bytes spilled by dataset operators.\n                DataContext.enable_get_object_locations_for_metrics\n                must be set to True to report this metric', tag_keys=op_tags_keys)
        self.bytes_allocated = Gauge('data_allocated_bytes', description='Bytes allocated by dataset operators', tag_keys=op_tags_keys)
        self.bytes_freed = Gauge('data_freed_bytes', description='Bytes freed by dataset operators', tag_keys=op_tags_keys)
        self.bytes_current = Gauge('data_current_bytes', description='Bytes currently in memory store used by dataset operators', tag_keys=op_tags_keys)
        self.cpu_usage = Gauge('data_cpu_usage_cores', description='CPUs allocated to dataset operators', tag_keys=op_tags_keys)
        self.gpu_usage = Gauge('data_gpu_usage_cores', description='GPUs allocated to dataset operators', tag_keys=op_tags_keys)
        self.bytes_outputted = Gauge('data_output_bytes', description='Bytes outputted by dataset operators', tag_keys=op_tags_keys)
        self.rows_outputted = Gauge('data_output_rows', description='Rows outputted by dataset operators', tag_keys=op_tags_keys)
        self.block_generation_time = Gauge('data_block_generation_seconds', description='Time spent generating blocks.', tag_keys=op_tags_keys)
        iter_tag_keys = ('dataset',)
        self.iter_total_blocked_s = Gauge('data_iter_total_blocked_seconds', description='Seconds user thread is blocked by iter_batches()', tag_keys=iter_tag_keys)
        self.iter_user_s = Gauge('data_iter_user_seconds', description='Seconds spent in user code', tag_keys=iter_tag_keys)

    def record_start(self, stats_uuid):
        self.start_time[stats_uuid] = time.perf_counter()
        self.fifo_queue.append(stats_uuid)
        if len(self.fifo_queue) > self.max_stats:
            uuid = self.fifo_queue.pop(0)
            if uuid in self.start_time:
                del self.start_time[uuid]
            if uuid in self.last_time:
                del self.last_time[uuid]
            if uuid in self.metadata:
                del self.metadata[uuid]

    def record_task(self, stats_uuid: str, task_idx: int, blocks_metadata: List[BlockMetadata]):
        for metadata in blocks_metadata:
            metadata.schema = None
        if stats_uuid in self.start_time:
            self.metadata[stats_uuid][task_idx] = blocks_metadata
            self.last_time[stats_uuid] = time.perf_counter()

    def get(self, stats_uuid):
        if stats_uuid not in self.metadata:
            return ({}, 0.0)
        return (self.metadata[stats_uuid], self.last_time[stats_uuid] - self.start_time[stats_uuid])

    def _get_stats_dict_size(self):
        return (len(self.start_time), len(self.last_time), len(self.metadata))

    def get_dataset_id(self):
        dataset_id = str(self.next_dataset_id)
        self.next_dataset_id += 1
        return dataset_id

    def update_metrics(self, execution_metrics, iteration_metrics):
        for metrics in execution_metrics:
            self.update_execution_metrics(*metrics)
        for metrics in iteration_metrics:
            self.update_iteration_metrics(*metrics)

    def update_execution_metrics(self, dataset_tag: str, op_metrics: List[Dict[str, Union[int, float]]], operator_tags: List[str], state: Dict[str, Any]):
        for stats, operator_tag in zip(op_metrics, operator_tags):
            tags = self._create_tags(dataset_tag, operator_tag)
            self.bytes_spilled.set(stats.get('obj_store_mem_spilled', 0), tags)
            self.bytes_allocated.set(stats.get('obj_store_mem_alloc', 0), tags)
            self.bytes_freed.set(stats.get('obj_store_mem_freed', 0), tags)
            self.bytes_current.set(stats.get('obj_store_mem_cur', 0), tags)
            self.bytes_outputted.set(stats.get('bytes_outputs_generated', 0), tags)
            self.rows_outputted.set(stats.get('rows_outputs_generated', 0), tags)
            self.cpu_usage.set(stats.get('cpu_usage', 0), tags)
            self.gpu_usage.set(stats.get('gpu_usage', 0), tags)
            self.block_generation_time.set(stats.get('block_generation_time', 0), tags)
        self.update_dataset(dataset_tag, state)

    def update_iteration_metrics(self, stats: 'DatasetStats', dataset_tag):
        tags = self._create_tags(dataset_tag)
        self.iter_total_blocked_s.set(stats.iter_total_blocked_s.get(), tags)
        self.iter_user_s.set(stats.iter_user_s.get(), tags)

    def clear_execution_metrics(self, dataset_tag: str, operator_tags: List[str]):
        for operator_tag in operator_tags:
            tags = self._create_tags(dataset_tag, operator_tag)
            self.bytes_spilled.set(0, tags)
            self.bytes_allocated.set(0, tags)
            self.bytes_freed.set(0, tags)
            self.bytes_current.set(0, tags)
            self.bytes_outputted.set(0, tags)
            self.rows_outputted.set(0, tags)
            self.cpu_usage.set(0, tags)
            self.gpu_usage.set(0, tags)
            self.block_generation_time.set(0, tags)

    def clear_iteration_metrics(self, dataset_tag: str):
        tags = self._create_tags(dataset_tag)
        self.iter_total_blocked_s.set(0, tags)
        self.iter_user_s.set(0, tags)

    def register_dataset(self, dataset_tag: str, operator_tags: List[str]):
        self.datasets[dataset_tag] = {'state': 'RUNNING', 'progress': 0, 'total': 0, 'start_time': time.time(), 'end_time': None, 'operators': {operator: {'state': 'RUNNING', 'progress': 0, 'total': 0} for operator in operator_tags}}

    def update_dataset(self, dataset_tag, state):
        self.datasets[dataset_tag].update(state)

    def get_datasets(self):
        return self.datasets

    def _create_tags(self, dataset_tag: str, operator_tag: Optional[str]=None):
        tags = {'dataset': dataset_tag}
        if operator_tag is not None:
            tags['operator'] = operator_tag
        return tags