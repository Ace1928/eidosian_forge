import functools
import sys
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple
import torch
from torch import fx
@functools.lru_cache(None)
def _lazy_import_entry_point(backend_name: str):
    from importlib.metadata import entry_points
    compiler_fn = None
    group_name = 'torch_dynamo_backends'
    if sys.version_info < (3, 10):
        backend_eps = entry_points()
        eps = [ep for ep in backend_eps.get(group_name, ()) if ep.name == backend_name]
        if len(eps) > 0:
            compiler_fn = eps[0].load()
    else:
        backend_eps = entry_points(group=group_name)
        if backend_name in backend_eps.names:
            compiler_fn = backend_eps[backend_name].load()
    if compiler_fn is not None and backend_name not in list_backends(tuple()):
        register_backend(compiler_fn=compiler_fn, name=backend_name)