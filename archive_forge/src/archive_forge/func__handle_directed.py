from __future__ import division
import numpy as np
from pygsp import utils
def _handle_directed(G):
    if not G.is_directed():
        return G
    else:
        from pygsp import graphs
        G2 = graphs.Graph(utils.symmetrize(G.W))
        G2.coords = G.coords
        G2.plotting = G.plotting
        return G2