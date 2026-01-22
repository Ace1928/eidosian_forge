from typing import TypedDict
class SphinxParallelSpec(TypedDict):
    parallel_read_safe: bool
    parallel_write_safe: bool