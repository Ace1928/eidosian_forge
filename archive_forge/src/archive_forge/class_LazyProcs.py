from functools import lru_cache, wraps
from typing import List, Optional, Dict
from asgiref.sync import async_to_sync
from .mp_utils import multiproc, _MAX_PROCS, lazy_parallelize
class LazyProcs:
    active = False
    active_procs: Dict[str, multiproc.Process] = {}
    inactive_procs: Dict[str, multiproc.Process] = {}

    @classmethod
    def set_state(cls):
        LazyProcs.active = bool(LazyProcs.num_active > 1)

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

    @classmethod
    def kill_proc(cls, name: str):
        if name in LazyProcs.active_procs:
            proc = LazyProcs.active_procs.pop(name)
            proc.terminate()
            LazyProcs.inactive_procs[name] = proc
            LazyProcs.set_state()

    @classmethod
    def killall(cls):
        for name in LazyProcs.active_procs:
            LazyProcs.kill_proc(name)

    @property
    def num_active(self):
        return len(self.active_procs)

    @property
    def num_inactive(self):
        return len(self.inactive_procs)