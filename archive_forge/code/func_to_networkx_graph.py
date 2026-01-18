import warnings
from collections.abc import Collection, Generator, Iterator
import networkx as nx
def to_networkx_graph(data, create_using=None, multigraph_input=False):
    """Make a NetworkX graph from a known data structure.

    The preferred way to call this is automatically
    from the class constructor

    >>> d = {0: {1: {"weight": 1}}}  # dict-of-dicts single edge (0,1)
    >>> G = nx.Graph(d)

    instead of the equivalent

    >>> G = nx.from_dict_of_dicts(d)

    Parameters
    ----------
    data : object to be converted

        Current known types are:
         any NetworkX graph
         dict-of-dicts
         dict-of-lists
         container (e.g. set, list, tuple) of edges
         iterator (e.g. itertools.chain) that produces edges
         generator of edges
         Pandas DataFrame (row per edge)
         2D numpy array
         scipy sparse array
         pygraphviz agraph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    multigraph_input : bool (default False)
        If True and  data is a dict_of_dicts,
        try to create a multigraph assuming dict_of_dict_of_lists.
        If data and create_using are both multigraphs then create
        a multigraph from a multigraph.

    """
    if hasattr(data, 'adj'):
        try:
            result = from_dict_of_dicts(data.adj, create_using=create_using, multigraph_input=data.is_multigraph())
            result.graph.update(data.graph)
            for n, dd in data.nodes.items():
                result._node[n].update(dd)
            return result
        except Exception as err:
            raise nx.NetworkXError('Input is not a correct NetworkX graph.') from err
    if hasattr(data, 'is_strict'):
        try:
            return nx.nx_agraph.from_agraph(data, create_using=create_using)
        except Exception as err:
            raise nx.NetworkXError('Input is not a correct pygraphviz graph.') from err
    if isinstance(data, dict):
        try:
            return from_dict_of_dicts(data, create_using=create_using, multigraph_input=multigraph_input)
        except Exception as err1:
            if multigraph_input is True:
                raise nx.NetworkXError(f'converting multigraph_input raised:\n{type(err1)}: {err1}')
            try:
                return from_dict_of_lists(data, create_using=create_using)
            except Exception as err2:
                raise TypeError('Input is not known type.') from err2
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            if data.shape[0] == data.shape[1]:
                try:
                    return nx.from_pandas_adjacency(data, create_using=create_using)
                except Exception as err:
                    msg = 'Input is not a correct Pandas DataFrame adjacency matrix.'
                    raise nx.NetworkXError(msg) from err
            else:
                try:
                    return nx.from_pandas_edgelist(data, edge_attr=True, create_using=create_using)
                except Exception as err:
                    msg = 'Input is not a correct Pandas DataFrame edge-list.'
                    raise nx.NetworkXError(msg) from err
    except ImportError:
        warnings.warn('pandas not found, skipping conversion test.', ImportWarning)
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            try:
                return nx.from_numpy_array(data, create_using=create_using)
            except Exception as err:
                raise nx.NetworkXError(f'Failed to interpret array as an adjacency matrix.') from err
    except ImportError:
        warnings.warn('numpy not found, skipping conversion test.', ImportWarning)
    try:
        import scipy
        if hasattr(data, 'format'):
            try:
                return nx.from_scipy_sparse_array(data, create_using=create_using)
            except Exception as err:
                raise nx.NetworkXError('Input is not a correct scipy sparse array type.') from err
    except ImportError:
        warnings.warn('scipy not found, skipping conversion test.', ImportWarning)
    if isinstance(data, (Collection, Generator, Iterator)):
        try:
            return from_edgelist(data, create_using=create_using)
        except Exception as err:
            raise nx.NetworkXError('Input is not a valid edge list') from err
    raise nx.NetworkXError('Input is not a known data type for conversion.')