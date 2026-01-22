from __future__ import annotations
import operator
import numpy as np
from dask import config, core
from dask.blockwise import Blockwise, fuse_roots, optimize_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, fuse
from dask.utils import ensure_dict
def _is_selection(layer):
    if not isinstance(layer, Blockwise):
        return False
    if layer.dsk[layer.output][0] != operator.getitem:
        return False
    return True