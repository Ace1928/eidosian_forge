import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup
def _is_supported_device(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda or tensor.device.type in ('xla', 'cpu')