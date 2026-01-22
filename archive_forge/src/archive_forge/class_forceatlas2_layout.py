from __future__ import annotations
import numpy as np
import param
import scipy.sparse
class forceatlas2_layout(LayoutAlgorithm):
    """
    Assign coordinates to the nodes using force-directed algorithm.

    This is a force-directed graph layout algorithm called
    `ForceAtlas2`.

    Timothee Poisot's `nxfa2` is the original implementation of this
    algorithm.

    .. _ForceAtlas2:
       http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0098679&type=printable
    .. _nxfa2:
       https://github.com/tpoisot/nxfa2
    """
    iterations = param.Integer(default=10, bounds=(1, None), doc='\n        Number of passes for the layout algorithm')
    linlog = param.Boolean(False, doc='\n        Whether to use logarithmic attraction force')
    nohubs = param.Boolean(False, doc='\n        Whether to grant authorities (nodes with a high indegree) a\n        more central position than hubs (nodes with a high outdegree)')
    k = param.Number(default=None, doc='\n        Compensates for the repulsion for nodes that are far away\n        from the center. Defaults to the inverse of the number of\n        nodes.')
    dim = param.Integer(default=2, bounds=(1, None), doc='\n        Coordinate dimensions of each node')

    def __call__(self, nodes, edges, **params):
        p = param.ParamOverrides(self, params)
        np.random.seed(p.seed)
        points = _extract_points_from_nodes(nodes, p, dtype='f')
        matrix = _convert_graph_to_sparse_matrix(nodes, edges, p, dtype='f')
        if p.k is None:
            p.k = np.sqrt(1.0 / len(points))
        temperature = 0.1
        cooling(matrix, points, temperature, p)
        return _merge_points_with_nodes(nodes, points, p)