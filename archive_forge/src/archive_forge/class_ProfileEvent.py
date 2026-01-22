import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes
@dataclasses.dataclass
class ProfileEvent:
    category: str
    key: str
    self_cuda_time_ms: float
    count: float