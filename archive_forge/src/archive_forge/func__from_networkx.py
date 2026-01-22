from collections import defaultdict
import numpy as np
import networkx as nx
import holoviews as _hv
from bokeh.models import HoverTool
from holoviews import Graph, Labels, dim
from holoviews.core.options import Store
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh import GraphPlot, LabelsPlot
from holoviews.plotting.bokeh.styles import markers
from .backend_transforms import _transfer_opts_cur_backend
from .util import process_crs
from .utilities import save, show # noqa
def _from_networkx(G, positions, nodes=None, cls=Graph, **kwargs):
    """
    Generate a Graph element from a networkx.Graph object and networkx
    layout function or dictionary of node positions.  Any keyword
    arguments will be passed to the layout function. By default it
    will extract all node and edge attributes from the networkx.Graph
    but explicit node information may also be supplied. Any non-scalar
    attributes, such as lists or dictionaries will be ignored.

    Parameters
    ----------
    G : networkx.Graph
       Graph to convert to Graph element
    positions : dict or callable
       Node positions defined as a dictionary mapping from node id to
       (x, y) tuple or networkx layout function which computes a
       positions dictionary.
    kwargs : dict
       Keyword arguments for the element

    Returns
    -------
    graph : holoviews.Graph
       Graph element
    """
    edges = defaultdict(list)
    for start, end in G.edges():
        for attr, value in sorted(G.adj[start][end].items()):
            if isinstance(value, (list, dict)):
                continue
            edges[attr].append(value)
        if isinstance(start, tuple):
            start = str(start)
        if isinstance(end, tuple):
            end = str(end)
        edges['start'].append(start)
        edges['end'].append(end)
    edge_cols = sorted((k for k in edges if k not in ('start', 'end') and len(edges[k]) == len(edges['start'])))
    edge_vdims = [str(col) if isinstance(col, int) else col for col in edge_cols]
    edge_data = tuple((edges[col] for col in ['start', 'end'] + edge_cols))
    xdim, ydim, idim = cls.node_type.kdims[:3]
    if nodes:
        node_columns = nodes.columns()
        idx_dim = nodes.kdims[0].name
        info_cols, values = zip(*((k, v) for k, v in node_columns.items() if k != idx_dim))
        node_info = {i: vals for i, vals in zip(node_columns[idx_dim], zip(*values))}
    else:
        info_cols = []
        node_info = None
    node_columns = defaultdict(list)
    for idx, pos in positions.items():
        node = G.nodes.get(idx)
        if node is None:
            continue
        x, y = pos
        node_columns[xdim.name].append(x)
        node_columns[ydim.name].append(y)
        for attr, value in node.items():
            if isinstance(value, (list, dict, tuple)):
                continue
            node_columns[attr].append(value)
        for i, col in enumerate(info_cols):
            node_columns[col].append(node_info[idx][i])
        if isinstance(idx, tuple):
            idx = str(idx)
        node_columns[idim.name].append(idx)
    node_cols = sorted((k for k in node_columns if k not in cls.node_type.kdims and len(node_columns[k]) == len(node_columns[xdim.name])))
    columns = [xdim.name, ydim.name, idim.name] + node_cols + list(info_cols)
    node_data = tuple((node_columns[col] for col in columns))
    vdims = []
    for col in node_cols:
        if isinstance(col, int):
            dim = str(col)
        elif nodes is not None and col in nodes.vdims:
            dim = nodes.get_dimension(col)
        else:
            dim = col
        vdims.append(dim)
    nodes = cls.node_type(node_data, vdims=vdims)
    return cls((edge_data, nodes), vdims=edge_vdims)