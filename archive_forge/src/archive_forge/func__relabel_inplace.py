import networkx as nx
def _relabel_inplace(G, mapping):
    if len(mapping.keys() & mapping.values()) > 0:
        D = nx.DiGraph(list(mapping.items()))
        D.remove_edges_from(nx.selfloop_edges(D))
        try:
            nodes = reversed(list(nx.topological_sort(D)))
        except nx.NetworkXUnfeasible as err:
            raise nx.NetworkXUnfeasible('The node label sets are overlapping and no ordering can resolve the mapping. Use copy=True.') from err
    else:
        nodes = [n for n in G if n in mapping]
    multigraph = G.is_multigraph()
    directed = G.is_directed()
    for old in nodes:
        try:
            new = mapping[old]
            G.add_node(new, **G.nodes[old])
        except KeyError:
            continue
        if new == old:
            continue
        if multigraph:
            new_edges = [(new, new if old == target else target, key, data) for _, target, key, data in G.edges(old, data=True, keys=True)]
            if directed:
                new_edges += [(new if old == source else source, new, key, data) for source, _, key, data in G.in_edges(old, data=True, keys=True)]
            seen = set()
            for i, (source, target, key, data) in enumerate(new_edges):
                if target in G[source] and key in G[source][target]:
                    new_key = 0 if not isinstance(key, (int, float)) else key
                    while new_key in G[source][target] or (target, new_key) in seen:
                        new_key += 1
                    new_edges[i] = (source, target, new_key, data)
                    seen.add((target, new_key))
        else:
            new_edges = [(new, new if old == target else target, data) for _, target, data in G.edges(old, data=True)]
            if directed:
                new_edges += [(new if old == source else source, new, data) for source, _, data in G.in_edges(old, data=True)]
        G.remove_node(old)
        G.add_edges_from(new_edges)
    return G