import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def underlying_graph(link):
    G = nx.Graph()
    edges = [(c, adj) for c in link.crossings for adj in map(lambda x: x[0], c.adjacent)]
    G.add_edges_from(edges)
    return G