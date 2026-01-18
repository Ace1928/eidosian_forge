from io import StringIO
from typing import Dict, List
import ray
from ray.data.context import DataContext
def trace_deallocation(ref: ray.ObjectRef, loc: str, free: bool=True) -> None:
    """Record that an object has been deleted (and delete if free=True).

    Args:
        ref: The object we no longer need.
        loc: A human-readable string identifying the call site.
        free: Whether to eagerly destroy the object instead of waiting for Ray
            reference counting to kick in.
    """
    if free:
        ray._private.internal_api.free(ref, local_only=False)
    ctx = DataContext.get_current()
    if ctx.trace_allocations:
        tracer = _get_mem_actor()
        ray.get(tracer.trace_dealloc.remote([ref], loc, free))