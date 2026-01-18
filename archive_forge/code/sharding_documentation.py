from typing import (
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from enum import IntEnum

    Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``).

    After ``apply_sharding`` is called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
    original DataPipe, where `n` equals to the number of instances.

    Args:
        source_datapipe: Iterable DataPipe that will be sharded
    