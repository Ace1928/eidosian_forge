from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad
from typing_extensions import TypeAlias
def _get_fused_kernels_supported_devices() -> List[str]:
    """Return the device type list that supports fused kernels in optimizer."""
    return ['cuda', 'xpu', torch._C._get_privateuse1_backend_name()]