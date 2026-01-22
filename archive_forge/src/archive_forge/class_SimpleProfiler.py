import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterator, List, Set, Tuple
import torch
import torch.distributed.fsdp._flat_param as flat_param_file
from torch.distributed.fsdp._common_utils import (
class SimpleProfiler:

    class Type(str, Enum):
        ALL = 'all'
        ALLGATHER = 'all_gather'
        ALLGATHER_OBJ = 'all_gather_object'
        RESHARDING = 'resharding'
        H2D = 'H2D'
        D2H = 'D2H'
    results: Dict[str, float] = defaultdict(float)
    profiling: Set[str] = set()

    @classmethod
    def reset(cls) -> None:
        cls.results.clear()
        cls.profiling.clear()

    @classmethod
    @contextmanager
    def profile(cls, profile_type: str) -> Iterator[None]:
        assert profile_type not in cls.profiling, f'{profile_type} is already being profiled. SimpleProfiler does not support profiling multiple instances at the same time. '
        cls.profiling.add(profile_type)
        begin = time.monotonic()
        try:
            yield
        finally:
            end = time.monotonic()
            cls.results[profile_type] += end - begin
            cls.profiling.remove(profile_type)

    @classmethod
    def dump_and_reset(cls, msg: str) -> None:
        logger.warning('%s %s', msg, str(cls.results))
        cls.reset()