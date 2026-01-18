from itertools import chain, repeat
import networkx as nx
def yield_relabeled(graphs):
    first_label = 0
    for G in graphs:
        yield nx.convert_node_labels_to_integers(G, first_label=first_label)
        first_label += len(G)