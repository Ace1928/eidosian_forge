import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup
class _GeneralMultiDeviceReplicator(_MultiDeviceReplicator):
    """
    Lazily serves tensor to request device. This class extends
    _MultiDeviceReplicator to allow support for "cpu" as a device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        assert _is_supported_device(master_tensor)
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}