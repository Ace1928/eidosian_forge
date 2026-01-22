import networkx as nx
@nx._dispatch(graphs=None)
def adjacency_graph(data, directed=False, multigraph=True, attrs=_attrs):
    """Returns graph from adjacency data format.

    Parameters
    ----------
    data : dict
        Adjacency list formatted graph data

    directed : bool
        If True, and direction not specified in data, return a directed graph.

    multigraph : bool
        If True, and multigraph not specified in data, return a multigraph.

    attrs : dict
        A dictionary that contains two keys 'id' and 'key'. The corresponding
        values provide the attribute names for storing NetworkX-internal graph
        data. The values should be unique. Default value:
        :samp:`dict(id='id', key='key')`.

    Returns
    -------
    G : NetworkX graph
       A NetworkX graph object

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.Graph([(1, 2)])
    >>> data = json_graph.adjacency_data(G)
    >>> H = json_graph.adjacency_graph(data)

    Notes
    -----
    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    adjacency_graph, node_link_data, tree_data
    """
    multigraph = data.get('multigraph', multigraph)
    directed = data.get('directed', directed)
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()
    id_ = attrs['id']
    key = None if not multigraph else attrs['key']
    graph.graph = dict(data.get('graph', []))
    mapping = []
    for d in data['nodes']:
        node_data = d.copy()
        node = node_data.pop(id_)
        mapping.append(node)
        graph.add_node(node)
        graph.nodes[node].update(node_data)
    for i, d in enumerate(data['adjacency']):
        source = mapping[i]
        for tdata in d:
            target_data = tdata.copy()
            target = target_data.pop(id_)
            if not multigraph:
                graph.add_edge(source, target)
                graph[source][target].update(target_data)
            else:
                ky = target_data.pop(key, None)
                graph.add_edge(source, target, key=ky)
                graph[source][target][ky].update(target_data)
    return graph