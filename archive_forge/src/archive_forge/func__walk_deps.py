from __future__ import annotations
import operator
import numpy as np
from dask import config, core
from dask.blockwise import Blockwise, fuse_roots, optimize_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, fuse
from dask.utils import ensure_dict
def _walk_deps(dependents, key, success):
    if key == success:
        return True
    deps = dependents[key]
    if deps:
        return all((_walk_deps(dependents, dep, success) for dep in deps))
    else:
        return False