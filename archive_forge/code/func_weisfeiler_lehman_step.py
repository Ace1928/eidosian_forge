from collections import Counter, defaultdict
from hashlib import blake2b
import networkx as nx
def weisfeiler_lehman_step(G, labels, node_subgraph_hashes, edge_attr=None):
    """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        Appends the new hashed label to the dictionary of subgraph hashes
        originating from and indexed by each node in G
        """
    new_labels = {}
    for node in G.nodes():
        label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
        hashed_label = _hash_label(label, digest_size)
        new_labels[node] = hashed_label
        node_subgraph_hashes[node].append(hashed_label)
    return new_labels