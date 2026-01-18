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
def to_pygsp(self, **kwargs):
    """Convert to a PyGSP graph

        For use only when the user means to create the graph using
        the flag `use_pygsp=True`, and doesn't wish to recompute the kernel.
        Creates a graphtools.graphs.TraditionalGraph with a precomputed
        affinity matrix which also inherits from pygsp.graphs.Graph.

        Parameters
        ----------
        kwargs
            keyword arguments for graphtools.Graph

        Returns
        -------
        G : graphtools.base.PyGSPGraph, graphtools.graphs.TraditionalGraph
        """
    from . import api
    if 'precomputed' in kwargs:
        if kwargs['precomputed'] != 'affinity':
            warnings.warn("Cannot build PyGSPGraph with precomputed={}. Using 'affinity' instead.".format(kwargs['precomputed']), UserWarning)
        del kwargs['precomputed']
    if 'use_pygsp' in kwargs:
        if kwargs['use_pygsp'] is not True:
            warnings.warn('Cannot build PyGSPGraph with use_pygsp={}. Use True instead.'.format(kwargs['use_pygsp']), UserWarning)
        del kwargs['use_pygsp']
    return api.Graph(self.K, precomputed='affinity', use_pygsp=True, **kwargs)