from typing import Dict, List, Optional, Union
import torch
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase
from . import constants as rpc_contants
def _to_device_list(devices: List[DeviceType]) -> List[torch.device]:
    return list(map(_to_device, devices))