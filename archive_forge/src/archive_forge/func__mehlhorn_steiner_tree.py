from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _mehlhorn_steiner_tree(G, terminal_nodes, weight):
    paths = nx.multi_source_dijkstra_path(G, terminal_nodes)
    d_1 = {}
    s = {}
    for v in G.nodes():
        s[v] = paths[v][0]
        d_1[v, s[v]] = len(paths[v]) - 1
    G_1_prime = nx.Graph()
    for u, v, data in G.edges(data=True):
        su, sv = (s[u], s[v])
        weight_here = d_1[u, su] + data.get(weight, 1) + d_1[v, sv]
        if not G_1_prime.has_edge(su, sv):
            G_1_prime.add_edge(su, sv, weight=weight_here)
        else:
            new_weight = min(weight_here, G_1_prime[su][sv][weight])
            G_1_prime.add_edge(su, sv, weight=new_weight)
    G_2 = nx.minimum_spanning_edges(G_1_prime, data=True)
    G_3 = nx.Graph()
    for u, v, d in G_2:
        path = nx.shortest_path(G, u, v, weight)
        for n1, n2 in pairwise(path):
            G_3.add_edge(n1, n2)
    G_3_mst = list(nx.minimum_spanning_edges(G_3, data=False))
    if G.is_multigraph():
        G_3_mst = ((u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for u, v in G_3_mst)
    G_4 = G.edge_subgraph(G_3_mst).copy()
    _remove_nonterminal_leaves(G_4, terminal_nodes)
    return G_4.edges()