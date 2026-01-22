from operator import itemgetter
import networkx as nx
def _node_betweenness(G, source, cutoff=False, normalized=True, weight=None):
    """Node betweenness_centrality helper:

    See betweenness_centrality for what you probably want.
    This actually computes "load" and not betweenness.
    See https://networkx.lanl.gov/ticket/103

    This calculates the load of each node for paths from a single source.
    (The fraction of number of shortests paths from source that go
    through each node.)

    To get the load for a node you need to do all-pairs shortest paths.

    If weight is not None then use Dijkstra for finding shortest paths.
    """
    if weight is None:
        pred, length = nx.predecessor(G, source, cutoff=cutoff, return_seen=True)
    else:
        pred, length = nx.dijkstra_predecessor_and_distance(G, source, cutoff, weight)
    onodes = [(l, vert) for vert, l in length.items()]
    onodes.sort()
    onodes[:] = [vert for l, vert in onodes if l > 0]
    between = {}.fromkeys(length, 1.0)
    while onodes:
        v = onodes.pop()
        if v in pred:
            num_paths = len(pred[v])
            for x in pred[v]:
                if x == source:
                    break
                between[x] += between[v] / num_paths
    for v in between:
        between[v] -= 1
    if normalized:
        l = len(between)
        if l > 2:
            scale = 1 / ((l - 1) * (l - 2))
            for v in between:
                between[v] *= scale
    return between