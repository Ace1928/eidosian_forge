import pickle
from typing import Optional, List, Tuple, TYPE_CHECKING
from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip, set_cuda_visible_devices, get_ip
class RayWorkerVllm:
    """Ray wrapper for vllm.worker.Worker, allowing Worker to be
        lazliy initialized after Ray sets CUDA_VISIBLE_DEVICES."""

    def __init__(self, init_cached_hf_modules=False) -> None:
        if init_cached_hf_modules:
            from transformers.dynamic_module_utils import init_hf_modules
            init_hf_modules()
        self.worker = None
        self.compiled_dag_cuda_device_set = False

    def init_worker(self, worker_init_fn):
        self.worker = worker_init_fn()

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def execute_method(self, method, *args, **kwargs):
        executor = getattr(self, method)
        return executor(*args, **kwargs)

    def get_node_ip(self) -> str:
        return get_ip()

    def get_node_and_gpu_ids(self) -> Tuple[str, List[int]]:
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_gpu_ids()
        return (node_id, gpu_ids)

    def set_cuda_visible_devices(self, device_ids) -> None:
        set_cuda_visible_devices(device_ids)

    def execute_model_compiled_dag_remote(self, ignored):
        """Used only when compiled DAG is enabled."""
        import torch
        if not self.compiled_dag_cuda_device_set:
            torch.cuda.set_device(self.worker.device)
            self.compiled_dag_cuda_device_set = True
        output = self.worker.execute_model()
        output = pickle.dumps(output)
        return output