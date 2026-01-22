import copy
from collections import defaultdict
import os
import time
import pickle
import importlib
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
from vllm.utils import (Counter, set_cuda_visible_devices, get_ip,
def _init_workers_ray(self, placement_group: 'PlacementGroup', **ray_remote_kwargs):
    if self.parallel_config.tensor_parallel_size == 1:
        num_gpus = self.cache_config.gpu_memory_utilization
    else:
        num_gpus = 1
    self.driver_dummy_worker: RayWorkerVllm = None
    self.workers: List[RayWorkerVllm] = []
    driver_ip = get_ip()
    for bundle_id, bundle in enumerate(placement_group.bundle_specs):
        if not bundle.get('GPU', 0):
            continue
        scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_capture_child_tasks=True, placement_group_bundle_index=bundle_id)
        worker = ray.remote(num_cpus=0, num_gpus=num_gpus, scheduling_strategy=scheduling_strategy, **ray_remote_kwargs)(RayWorkerVllm).remote(self.model_config.trust_remote_code)
        worker_ip = ray.get(worker.get_node_ip.remote())
        if worker_ip == driver_ip and self.driver_dummy_worker is None:
            self.driver_dummy_worker = worker
        else:
            self.workers.append(worker)
    if self.driver_dummy_worker is None:
        raise ValueError('Ray does not allocate any GPUs on the driver node. Consider adjusting the Ray placement group or running the driver on a GPU node.')
    driver_node_id, driver_gpu_ids = ray.get(self.driver_dummy_worker.get_node_and_gpu_ids.remote())
    worker_node_and_gpu_ids = ray.get([worker.get_node_and_gpu_ids.remote() for worker in self.workers])
    node_workers = defaultdict(list)
    node_gpus = defaultdict(list)
    node_workers[driver_node_id].append(0)
    node_gpus[driver_node_id].extend(driver_gpu_ids)
    for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids, start=1):
        node_workers[node_id].append(i)
        node_gpus[node_id].extend(gpu_ids)
    for node_id, gpu_ids in node_gpus.items():
        node_gpus[node_id] = sorted(gpu_ids)
    set_cuda_visible_devices(node_gpus[driver_node_id])
    for worker, (node_id, _) in zip(self.workers, worker_node_and_gpu_ids):
        worker.set_cuda_visible_devices.remote(node_gpus[node_id])
    distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())
    Worker = self._dispatch_worker()
    model_config = copy.deepcopy(self.model_config)
    parallel_config = copy.deepcopy(self.parallel_config)
    scheduler_config = copy.deepcopy(self.scheduler_config)
    device_config = copy.deepcopy(self.device_config)
    for rank, (worker, (node_id, _)) in enumerate(zip(self.workers, worker_node_and_gpu_ids), start=1):
        local_rank = node_workers[node_id].index(rank)
        worker.init_worker.remote(lambda rank=rank, local_rank=local_rank: Worker(model_config, parallel_config, scheduler_config, device_config, local_rank, rank, distributed_init_method, lora_config=self.lora_config, kv_cache_dtype=self.cache_config.cache_dtype))
    driver_rank = 0
    driver_local_rank = node_workers[driver_node_id].index(driver_rank)
    self.driver_worker = Worker(model_config, parallel_config, scheduler_config, device_config, driver_local_rank, driver_rank, distributed_init_method, lora_config=self.lora_config, kv_cache_dtype=self.cache_config.cache_dtype, is_driver_worker=True)
    self._run_workers('init_model', cupy_port=get_open_port() if not model_config.enforce_eager else None)
    self._run_workers('load_model', max_concurrent_workers=self.parallel_config.max_parallel_loading_workers)