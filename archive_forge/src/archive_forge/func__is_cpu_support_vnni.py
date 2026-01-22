from contextlib import AbstractContextManager
from typing import Any, Optional, Union
import torch
from .. import device as _device
from . import amp
def _is_cpu_support_vnni() -> bool:
    """Returns a bool indicating if CPU supports VNNI."""
    return torch._C._cpu._is_cpu_support_vnni()