from enum import Enum
from dataclasses import dataclass
from datetime import timedelta
@dataclass
class SendOptions:
    dst_rank = 0
    dst_gpu_index = 0
    n_elements = 0
    timeout_ms = unset_timeout_ms