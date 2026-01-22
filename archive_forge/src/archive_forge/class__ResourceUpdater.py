import logging
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, Optional
import ray
from ray._private.resource_spec import NODE_ID_PREFIX
class _ResourceUpdater:
    """Periodic Resource updater for Tune.

    Initially, all resources are set to 0. The updater will try to update resources
    when (1) init ResourceUpdater (2) call "update_avail_resources", "num_cpus"
    or "num_gpus".

    The update takes effect when (1) Ray is initialized (2) the interval between
    this and last update is larger than "refresh_period"
    """

    def __init__(self, refresh_period: Optional[float]=None):
        self._avail_resources = _Resources(cpu=0, gpu=0)
        if refresh_period is None:
            refresh_period = float(os.environ.get('TUNE_STATE_REFRESH_PERIOD', TUNE_STATE_REFRESH_PERIOD))
        self._refresh_period = refresh_period
        self._last_resource_refresh = float('-inf')
        self.update_avail_resources()

    def update_avail_resources(self, num_retries: int=5, force: bool=False):
        if not ray.is_initialized():
            return
        if time.time() - self._last_resource_refresh < self._refresh_period and (not force):
            return
        logger.debug('Checking Ray cluster resources.')
        resources = None
        for i in range(num_retries):
            if i > 0:
                logger.warning(f'Cluster resources not detected or are 0. Attempt #{i + 1}...')
                time.sleep(0.5)
            resources = ray.cluster_resources()
            if resources:
                break
        if not resources:
            resources.setdefault('CPU', 0)
            resources.setdefault('GPU', 0)
            logger.warning('Cluster resources cannot be detected or are 0. You can resume this experiment by passing in `resume=True` to `run`.')
        resources = resources.copy()
        num_cpus = resources.pop('CPU', 0)
        num_gpus = resources.pop('GPU', 0)
        memory = resources.pop('memory', 0)
        object_store_memory = resources.pop('object_store_memory', 0)
        custom_resources = resources
        self._avail_resources = _Resources(int(num_cpus), int(num_gpus), memory=int(memory), object_store_memory=int(object_store_memory), custom_resources=custom_resources)
        self._last_resource_refresh = time.time()

    def _get_used_avail_resources(self, total_allocated_resources: Dict[str, Any]):
        total_allocated_resources = total_allocated_resources.copy()
        used_cpu = total_allocated_resources.pop('CPU', 0)
        total_cpu = self._avail_resources.cpu
        used_gpu = total_allocated_resources.pop('GPU', 0)
        total_gpu = self._avail_resources.gpu
        custom_used_total = {name: (total_allocated_resources.get(name, 0.0), self._avail_resources.get_res_total(name)) for name in self._avail_resources.custom_resources if not name.startswith(NODE_ID_PREFIX) and (total_allocated_resources.get(name, 0.0) > 0 or '_group_' not in name)}
        return (used_cpu, total_cpu, used_gpu, total_gpu, custom_used_total)

    def debug_string(self, total_allocated_resources: Dict[str, Any]) -> str:
        """Returns a human readable message for printing to the console."""
        if self._last_resource_refresh > 0:
            used_cpu, total_cpu, used_gpu, total_gpu, custom_used_total = self._get_used_avail_resources(total_allocated_resources)
            if used_cpu > total_cpu or used_gpu > total_gpu or any((used > total for used, total in custom_used_total.values())):
                self.update_avail_resources(force=True)
                used_cpu, total_cpu, used_gpu, total_gpu, custom_used_total = self._get_used_avail_resources(total_allocated_resources)
            status = f'Logical resource usage: {used_cpu}/{total_cpu} CPUs, {used_gpu}/{total_gpu} GPUs'
            customs = ', '.join((f'{used}/{total} {name}' for name, (used, total) in custom_used_total.items()))
            if customs:
                status += f' ({customs})'
            return status
        else:
            return 'Logical resource usage: ?'

    def get_num_cpus(self) -> int:
        self.update_avail_resources()
        return self._avail_resources.cpu

    def get_num_gpus(self) -> int:
        self.update_avail_resources()
        return self._avail_resources.gpu

    def __reduce__(self):
        return (_ResourceUpdater, (self._refresh_period,))