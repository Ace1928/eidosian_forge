import networkx as nx
from networkx.exception import NetworkXAlgorithmError
from networkx.utils import not_implemented_for
def weight_function(G, u, v):
    return len(set(G[u]) & set(pred[v]))