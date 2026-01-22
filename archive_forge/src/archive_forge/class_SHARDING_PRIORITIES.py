from typing import (
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from enum import IntEnum
class SHARDING_PRIORITIES(IntEnum):
    DEFAULT = 1
    DISTRIBUTED = 2
    MULTIPROCESSING = 3