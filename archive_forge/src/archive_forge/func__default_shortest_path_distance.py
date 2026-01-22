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
def _default_shortest_path_distance(self):
    if not self.weighted:
        distance = 'data'
        _logger.log_info('Using ambient data distances.')
    else:
        distance = 'affinity'
        _logger.log_info('Using negative log affinity distances.')
    return distance