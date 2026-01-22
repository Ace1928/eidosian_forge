from typing import Any
import torch
import enum
from torch._C import _from_dlpack
from torch._C import _to_dlpack as to_dlpack
class DLDeviceType(enum.IntEnum):
    kDLCPU = (1,)
    kDLGPU = (2,)
    kDLCPUPinned = (3,)
    kDLOpenCL = (4,)
    kDLVulkan = (7,)
    kDLMetal = (8,)
    kDLVPI = (9,)
    kDLROCM = (10,)
    kDLExtDev = (12,)
    kDLOneAPI = (14,)