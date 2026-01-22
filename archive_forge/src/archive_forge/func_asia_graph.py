from itertools import combinations
import pytest
import networkx as nx
def asia_graph():
    """Return the 'Asia' PGM graph."""
    G = nx.DiGraph(name='asia')
    G.add_edges_from([('asia', 'tuberculosis'), ('smoking', 'cancer'), ('smoking', 'bronchitis'), ('tuberculosis', 'either'), ('cancer', 'either'), ('either', 'xray'), ('either', 'dyspnea'), ('bronchitis', 'dyspnea')])
    nx.freeze(G)
    return G