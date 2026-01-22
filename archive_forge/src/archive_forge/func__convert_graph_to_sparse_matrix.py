from __future__ import annotations
import numpy as np
import param
import scipy.sparse
def _convert_graph_to_sparse_matrix(nodes, edges, params, dtype=None, format='csr'):
    nlen = len(nodes)
    if params.id is not None and params.id in nodes:
        index = dict(zip(nodes[params.id].values, range(nlen)))
    else:
        index = dict(zip(nodes.index.values, range(nlen)))
    if params.weight and params.weight in edges:
        edge_values = edges[[params.source, params.target, params.weight]].values
        rows, cols, data = zip(*((index[src], index[dst], weight) for src, dst, weight in edge_values if src in index and dst in index))
    else:
        edge_values = edges[[params.source, params.target]].values
        rows, cols, data = zip(*((index[src], index[dst], 1) for src, dst in edge_values if src in index and dst in index))
    d = data + data
    r = rows + cols
    c = cols + rows
    loops = edges[edges[params.source] == edges[params.target]]
    if len(loops):
        if params.weight and params.weight in edges:
            loop_values = loops[[params.source, params.target, params.weight]].values
            diag_index, diag_data = zip(*((index[src], -weight) for src, dst, weight in loop_values if src in index and dst in index))
        else:
            loop_values = loops[[params.source, params.target]].values
            diag_index, diag_data = zip(*((index[src], -1) for src, dst in loop_values if src in index and dst in index))
        d += diag_data
        r += diag_index
        c += diag_index
    M = scipy.sparse.coo_matrix((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
    return M.asformat(format)