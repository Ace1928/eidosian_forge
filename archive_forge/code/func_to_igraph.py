from . import matrix
from . import utils
from builtins import super
from copy import copy as shallow_copy
from future.utils import with_metaclass
from inspect import signature
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import abc
import numbers
import numpy as np
import pickle
import pygsp
import sys
import tasklogger
import warnings
def to_igraph(self, attribute='weight', **kwargs):
    """Convert to an igraph Graph

        Uses the igraph.Graph constructor

        Parameters
        ----------
        attribute : str, optional (default: "weight")
        kwargs : additional arguments for igraph.Graph
        """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError('Please install igraph with `pip install --user python-igraph`.')
    try:
        W = self.W
    except AttributeError:
        W = self.K.copy()
        W = matrix.set_diagonal(W, 0)
    sources, targets = W.nonzero()
    edgelist = list(zip(sources, targets))
    g = ig.Graph(W.shape[0], edgelist, **kwargs)
    weights = W[W.nonzero()]
    weights = matrix.to_array(weights)
    g.es[attribute] = weights.flatten().tolist()
    return g