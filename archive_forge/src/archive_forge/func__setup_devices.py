from dataclasses import dataclass, field
from typing import Tuple
from ..utils import cached_property, is_torch_available, is_torch_tpu_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments
@cached_property
def _setup_devices(self) -> Tuple['torch.device', int]:
    requires_backends(self, ['torch'])
    logger.info('PyTorch: setting up devices')
    if not self.cuda:
        device = torch.device('cpu')
        n_gpu = 0
    elif is_torch_tpu_available():
        device = xm.xla_device()
        n_gpu = 0
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count()
    return (device, n_gpu)