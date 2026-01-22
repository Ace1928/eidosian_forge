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
def _run_workers(self, method: str, *args, driver_args: Optional[List[Any]]=None, driver_kwargs: Optional[Dict[str, Any]]=None, max_concurrent_workers: Optional[int]=None, use_ray_compiled_dag: bool=False, **kwargs) -> Any:
    """Runs the given method on all workers."""
    if max_concurrent_workers:
        raise NotImplementedError('max_concurrent_workers is not supported yet.')
    if use_ray_compiled_dag:
        output_channels = self.forward_dag.execute(1)
    else:
        ray_worker_outputs = [worker.execute_method.remote(method, *args, **kwargs) for worker in self.workers]
    if driver_args is None:
        driver_args = args
    if driver_kwargs is None:
        driver_kwargs = kwargs
    driver_worker_output = getattr(self.driver_worker, method)(*driver_args, **driver_kwargs)
    if self.workers:
        if use_ray_compiled_dag:
            try:
                ray_worker_outputs = [pickle.loads(chan.begin_read()) for chan in output_channels]
            finally:
                for chan in output_channels:
                    chan.end_read()
        else:
            ray_worker_outputs = ray.get(ray_worker_outputs)
    return [driver_worker_output] + ray_worker_outputs