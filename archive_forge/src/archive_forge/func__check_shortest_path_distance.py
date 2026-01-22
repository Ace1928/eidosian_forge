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
def _check_shortest_path_distance(self, distance):
    if distance == 'data' and self.weighted:
        raise NotImplementedError("Graph shortest path with constant or data distance only implemented for unweighted graphs. For weighted graphs, use `distance='affinity'`.")
    elif distance == 'constant' and self.weighted:
        raise NotImplementedError("Graph shortest path with constant distance only implemented for unweighted graphs. For weighted graphs, use `distance='affinity'`.")
    elif distance == 'affinity' and (not self.weighted):
        raise ValueError("Graph shortest path with affinity distance only valid for weighted graphs. For unweighted graphs, use `distance='constant'` or `distance='data'`.")