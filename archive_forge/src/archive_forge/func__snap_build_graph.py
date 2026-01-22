from collections import Counter, defaultdict
import networkx as nx
def _snap_build_graph(G, groups, node_attributes, edge_attributes, neighbor_info, edge_types, prefix, supernode_attribute, superedge_attribute):
    """
    Build the summary graph from the data structures produced in the SNAP aggregation algorithm

    Used in the SNAP aggregation algorithm to build the output summary graph and supernode
    lookup dictionary.  This process uses the original graph and the data structures to
    create the supernodes with the correct node attributes, and the superedges with the correct
    edge attributes

    Parameters
    ----------
    G: networkx.Graph
        the original graph to be summarized
    groups: dict
        A dictionary of unique group IDs and their corresponding node groups
    node_attributes: iterable
        An iterable of the node attributes considered in the summarization process
    edge_attributes: iterable
        An iterable of the edge attributes considered in the summarization process
    neighbor_info: dict
        A data structure indicating the number of edges a node has with the
        groups in the current summarization of each edge type
    edge_types: dict
        dictionary of edges in the graph and their corresponding attributes recognized
        in the summarization
    prefix: string
        The prefix to be added to all supernodes
    supernode_attribute: str
        The node attribute for recording the supernode groupings of nodes
    superedge_attribute: str
        The edge attribute for recording the edge types represented by superedges

    Returns
    -------
    summary graph: Networkx graph
    """
    output = G.__class__()
    node_label_lookup = {}
    for index, group_id in enumerate(groups):
        group_set = groups[group_id]
        supernode = f'{prefix}{index}'
        node_label_lookup[group_id] = supernode
        supernode_attributes = {attr: G.nodes[next(iter(group_set))][attr] for attr in node_attributes}
        supernode_attributes[supernode_attribute] = group_set
        output.add_node(supernode, **supernode_attributes)
    for group_id in groups:
        group_set = groups[group_id]
        source_supernode = node_label_lookup[group_id]
        for other_group, group_edge_types in neighbor_info[next(iter(group_set))].items():
            if group_edge_types:
                target_supernode = node_label_lookup[other_group]
                summary_graph_edge = (source_supernode, target_supernode)
                edge_types = [dict(zip(edge_attributes, edge_type)) for edge_type in group_edge_types]
                has_edge = output.has_edge(*summary_graph_edge)
                if output.is_multigraph():
                    if not has_edge:
                        for edge_type in edge_types:
                            output.add_edge(*summary_graph_edge, **edge_type)
                    elif not output.is_directed():
                        existing_edge_data = output.get_edge_data(*summary_graph_edge)
                        for edge_type in edge_types:
                            if edge_type not in existing_edge_data.values():
                                output.add_edge(*summary_graph_edge, **edge_type)
                else:
                    superedge_attributes = {superedge_attribute: edge_types}
                    output.add_edge(*summary_graph_edge, **superedge_attributes)
    return output