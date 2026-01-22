from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
class connect_edges(param.ParameterizedFunction):
    """
    Convert a graph into paths suitable for datashading.

    Base class that connects each edge using a single line segment.
    Subclasses can add more complex algorithms for connecting with
    curved or manhattan-style polylines.
    """
    x = param.String(default='x', doc="\n        Column name for each node's x coordinate.")
    y = param.String(default='y', doc="\n        Column name for each node's y coordinate.")
    source = param.String(default='source', doc="\n        Column name for each edge's source.")
    target = param.String(default='target', doc="\n        Column name for each edge's target.")
    weight = param.String(default=None, allow_None=True, doc='\n        Column name for each edge weight. If None, weights are ignored.')
    include_edge_id = param.Boolean(default=False, doc='\n        Include edge IDs in bundled dataframe')

    def __call__(self, nodes, edges, **params):
        """
        Convert a graph data structure into a path structure for plotting

        Given a set of nodes (as a dataframe with a unique ID for each
        node) and a set of edges (as a dataframe with with columns for the
        source and destination IDs for each edge), returns a dataframe
        with with one path for each edge suitable for use with
        Datashader. The returned dataframe has columns for x and y
        location, with paths represented as successive points separated by
        a point with NaN as the x or y value.
        """
        p = param.ParamOverrides(self, params)
        edges, segment_class = _convert_graph_to_edge_segments(nodes, edges, p)
        return _convert_edge_segments_to_dataframe(edges, segment_class, p)