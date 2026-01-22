from functools import lru_cache, wraps
from typing import List, Optional, Dict
from asgiref.sync import async_to_sync
from .mp_utils import multiproc, _MAX_PROCS, lazy_parallelize
@classmethod
def add_proc(cls, proc: LazyProc):
    process = multiproc.Process(**proc.config)
    if proc.start:
        process.start()
        LazyProcs.active_procs[proc.name] = process
    else:
        LazyProcs.inactive_procs[proc.name] = process
    LazyProcs.set_state()
    return process