from __future__ import annotations
import numpy as np
import param
import scipy.sparse
def _extract_points_from_nodes(nodes, params, dtype=None):
    if params.x in nodes.columns and params.y in nodes.columns:
        points = np.asarray(nodes[[params.x, params.y]])
    else:
        points = np.asarray(np.random.random((len(nodes), params.dim)), dtype=dtype)
    return points