import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes

        ev.self_cuda_time_total is in microsecond. Convert to millisecond.
        