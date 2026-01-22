from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
def _inner_bellman_ford(G, sources, weight, pred, dist=None, heuristic=True):
    """Inner Relaxation loop for Bellman–Ford algorithm.

    This is an implementation of the SPFA variant.
    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

    Parameters
    ----------
    G : NetworkX graph

    source: list
        List of source nodes. The shortest path from any of the source
        nodes will be found if multiple sources are provided.

    weight : function
        The weight of an edge is the value returned by the function. The
        function must accept exactly three positional arguments: the two
        endpoints of an edge and the dictionary of edge attributes for
        that edge. The function must return a number.

    pred: dict of lists
        dict to store a list of predecessors keyed by that node

    dist: dict, optional (default=None)
        dict to store distance from source to the keyed node
        If None, returned dist dict contents default to 0 for every node in the
        source list

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    node or None
        Return a node `v` where processing discovered a negative cycle.
        If no negative cycle found, return None.

    Raises
    ------
    NodeNotFound
        If any of `source` is not in `G`.
    """
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f'Source {s} not in G')
    if pred is None:
        pred = {v: [] for v in sources}
    if dist is None:
        dist = {v: 0 for v in sources}
    nonexistent_edge = (None, None)
    pred_edge = {v: None for v in sources}
    recent_update = {v: nonexistent_edge for v in sources}
    G_succ = G._adj
    inf = float('inf')
    n = len(G)
    count = {}
    q = deque(sources)
    in_q = set(sources)
    while q:
        u = q.popleft()
        in_q.remove(u)
        if all((pred_u not in in_q for pred_u in pred[u])):
            dist_u = dist[u]
            for v, e in G_succ[u].items():
                dist_v = dist_u + weight(u, v, e)
                if dist_v < dist.get(v, inf):
                    if heuristic:
                        if v in recent_update[u]:
                            pred[v].append(u)
                            return v
                        if v in pred_edge and pred_edge[v] == u:
                            recent_update[v] = recent_update[u]
                        else:
                            recent_update[v] = (u, v)
                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0) + 1
                        if count_v == n:
                            return v
                        count[v] = count_v
                    dist[v] = dist_v
                    pred[v] = [u]
                    pred_edge[v] = u
                elif dist.get(v) is not None and dist_v == dist.get(v):
                    pred[v].append(u)
    return None