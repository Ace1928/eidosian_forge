from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def add_worker_processes(cls, name: str, procs: List['multiprocessing.Process'], kind: Optional[str]=None):
    """
        Adds worker processes
        """
    if kind is None:
        kind = 'default'
    if cls.worker_processes.get(kind) is None:
        cls.worker_processes[kind] = {}
    if cls.worker_processes[kind].get(name) is None:
        cls.worker_processes[kind][name] = []
    cls.worker_processes[kind][name].extend(procs)