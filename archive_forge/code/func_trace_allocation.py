from io import StringIO
from typing import Dict, List
import ray
from ray.data.context import DataContext
def trace_allocation(ref: ray.ObjectRef, loc: str) -> None:
    """Record that an object has been created.

    Args:
        ref: The object created.
        loc: A human-readable string identifying the call site.
    """
    ctx = DataContext.get_current()
    if ctx.trace_allocations:
        tracer = _get_mem_actor()
        ray.get(tracer.trace_alloc.remote([ref], loc))