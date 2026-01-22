import networkx as nx
import numpy as np
from scipy.sparse import linalg
from . import _ncut, _ncut_cy
def _label_all(rag, attr_name):
    """Assign a unique integer to the given attribute in the RAG.

    This function assumes that all labels in `rag` are unique. It
    picks up a random label from them and assigns it to the `attr_name`
    attribute of all the nodes.

    rag : RAG
        The Region Adjacency Graph.
    attr_name : string
        The attribute to which a unique integer is assigned.
    """
    node = min(rag.nodes())
    new_label = rag.nodes[node]['labels'][0]
    for n, d in rag.nodes(data=True):
        d[attr_name] = new_label