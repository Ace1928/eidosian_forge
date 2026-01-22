import sys
from .profiler import _Profiler, logger
class DCGMProfiler:
    """The dummy DCGM Profiler."""

    def __init__(self, main_profiler: '_Profiler', gpus_to_profile=None, field_ids_to_profile=None, updateFreq=None) -> None:
        pass

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def step(self) -> None:
        pass