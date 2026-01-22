import linecache
from typing import Any, List, Tuple, Optional
def _bpython_clear_linecache() -> None:
    if isinstance(linecache.cache, BPythonLinecache):
        bpython_history = linecache.cache.bpython_history
    else:
        bpython_history = None
    linecache.cache = BPythonLinecache(bpython_history)