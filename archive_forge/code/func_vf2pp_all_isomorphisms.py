import collections
import networkx as nx
@nx._dispatch(graphs={'G1': 0, 'G2': 1}, node_attrs={'node_label': 'default_label'})
def vf2pp_all_isomorphisms(G1, G2, node_label=None, default_label=None):
    """Yields all the possible mappings between G1 and G2.

    Parameters
    ----------
    G1, G2 : NetworkX Graph or MultiGraph instances.
        The two graphs to check for isomorphism.

    node_label : str, optional
        The name of the node attribute to be used when comparing nodes.
        The default is `None`, meaning node attributes are not considered
        in the comparison. Any node that doesn't have the `node_label`
        attribute uses `default_label` instead.

    default_label : scalar
        Default value to use when a node doesn't have an attribute
        named `node_label`. Default is `None`.

    Yields
    ------
    dict
        Isomorphic mapping between the nodes in `G1` and `G2`.
    """
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return False
    if G1.is_directed():
        G1_degree = {n: (in_degree, out_degree) for (n, in_degree), (_, out_degree) in zip(G1.in_degree, G1.out_degree)}
        G2_degree = {n: (in_degree, out_degree) for (n, in_degree), (_, out_degree) in zip(G2.in_degree, G2.out_degree)}
    else:
        G1_degree = dict(G1.degree)
        G2_degree = dict(G2.degree)
    if not G1.is_directed():
        find_candidates = _find_candidates
        restore_Tinout = _restore_Tinout
    else:
        find_candidates = _find_candidates_Di
        restore_Tinout = _restore_Tinout_Di
    if G1.order() != G2.order():
        return False
    if sorted(G1_degree.values()) != sorted(G2_degree.values()):
        return False
    graph_params, state_params = _initialize_parameters(G1, G2, G2_degree, node_label, default_label)
    if not _precheck_label_properties(graph_params):
        return False
    node_order = _matching_order(graph_params)
    stack = []
    candidates = iter(find_candidates(node_order[0], graph_params, state_params, G1_degree))
    stack.append((node_order[0], candidates))
    mapping = state_params.mapping
    reverse_mapping = state_params.reverse_mapping
    matching_node = 1
    while stack:
        current_node, candidate_nodes = stack[-1]
        try:
            candidate = next(candidate_nodes)
        except StopIteration:
            stack.pop()
            matching_node -= 1
            if stack:
                popped_node1, _ = stack[-1]
                popped_node2 = mapping[popped_node1]
                mapping.pop(popped_node1)
                reverse_mapping.pop(popped_node2)
                restore_Tinout(popped_node1, popped_node2, graph_params, state_params)
            continue
        if _feasibility(current_node, candidate, graph_params, state_params):
            if len(mapping) == G2.number_of_nodes() - 1:
                cp_mapping = mapping.copy()
                cp_mapping[current_node] = candidate
                yield cp_mapping
                continue
            mapping[current_node] = candidate
            reverse_mapping[candidate] = current_node
            _update_Tinout(current_node, candidate, graph_params, state_params)
            candidates = iter(find_candidates(node_order[matching_node], graph_params, state_params, G1_degree))
            stack.append((node_order[matching_node], candidates))
            matching_node += 1