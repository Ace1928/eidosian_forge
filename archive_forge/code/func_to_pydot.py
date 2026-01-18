import warnings
from locale import getpreferredencoding
import networkx as nx
from networkx.utils import open_file
def to_pydot(N):
    """Returns a pydot graph from a NetworkX graph N.

    Parameters
    ----------
    N : NetworkX graph
      A graph created with NetworkX

    Examples
    --------
    >>> K5 = nx.complete_graph(5)
    >>> P = nx.nx_pydot.to_pydot(K5)

    Notes
    -----

    """
    import pydot
    msg = 'nx.nx_pydot.to_pydot depends on the pydot package, which has known issues and is not actively maintained.\n\nSee https://github.com/networkx/networkx/issues/5723'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    if N.is_directed():
        graph_type = 'digraph'
    else:
        graph_type = 'graph'
    strict = nx.number_of_selfloops(N) == 0 and (not N.is_multigraph())
    name = N.name
    graph_defaults = N.graph.get('graph', {})
    if name == '':
        P = pydot.Dot('', graph_type=graph_type, strict=strict, **graph_defaults)
    else:
        P = pydot.Dot(f'"{name}"', graph_type=graph_type, strict=strict, **graph_defaults)
    try:
        P.set_node_defaults(**N.graph['node'])
    except KeyError:
        pass
    try:
        P.set_edge_defaults(**N.graph['edge'])
    except KeyError:
        pass
    for n, nodedata in N.nodes(data=True):
        str_nodedata = {str(k): str(v) for k, v in nodedata.items()}
        n = str(n)
        raise_error = _check_colon_quotes(n) or any((_check_colon_quotes(k) or _check_colon_quotes(v) for k, v in str_nodedata.items()))
        if raise_error:
            raise ValueError(f"""Node names and attributes should not contain ":" unless they are quoted with "".                For example the string 'attribute:data1' should be written as '"attribute:data1"'.                Please refer https://github.com/pydot/pydot/issues/258""")
        p = pydot.Node(n, **str_nodedata)
        P.add_node(p)
    if N.is_multigraph():
        for u, v, key, edgedata in N.edges(data=True, keys=True):
            str_edgedata = {str(k): str(v) for k, v in edgedata.items() if k != 'key'}
            u, v = (str(u), str(v))
            raise_error = _check_colon_quotes(u) or _check_colon_quotes(v) or any((_check_colon_quotes(k) or _check_colon_quotes(val) for k, val in str_edgedata.items()))
            if raise_error:
                raise ValueError(f"""Node names and attributes should not contain ":" unless they are quoted with "".                    For example the string 'attribute:data1' should be written as '"attribute:data1"'.                    Please refer https://github.com/pydot/pydot/issues/258""")
            edge = pydot.Edge(u, v, key=str(key), **str_edgedata)
            P.add_edge(edge)
    else:
        for u, v, edgedata in N.edges(data=True):
            str_edgedata = {str(k): str(v) for k, v in edgedata.items()}
            u, v = (str(u), str(v))
            raise_error = _check_colon_quotes(u) or _check_colon_quotes(v) or any((_check_colon_quotes(k) or _check_colon_quotes(val) for k, val in str_edgedata.items()))
            if raise_error:
                raise ValueError(f"""Node names and attributes should not contain ":" unless they are quoted with "".                    For example the string 'attribute:data1' should be written as '"attribute:data1"'.                    Please refer https://github.com/pydot/pydot/issues/258""")
            edge = pydot.Edge(u, v, **str_edgedata)
            P.add_edge(edge)
    return P