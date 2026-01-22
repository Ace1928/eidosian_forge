import pandas as pd
import graphtools.base
import graphtools
import pygsp
import scprep
import sklearn
def _check_pygsp_graph(G):
    if isinstance(G, graphtools.base.BaseGraph):
        if not isinstance(G, pygsp.graphs.Graph):
            G = G.to_pygsp()
    else:
        raise TypeError('Input graph should be of type graphtools.base.BaseGraph. With graphtools, use the `use_pygsp=True` flag.')
    return G