import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def allocate_vecattrs(self, attrs=None, types=None):
    """Ensure arrays for given per-vector extra-attribute names & types exist, at right size.

        The length of the index_to_key list is canonical 'intended size' of KeyedVectors,
        even if other properties (vectors array) hasn't yet been allocated or expanded.
        So this allocation targets that size.

        """
    if attrs is None:
        attrs = list(self.expandos.keys())
        types = [self.expandos[attr].dtype for attr in attrs]
    target_size = len(self.index_to_key)
    for attr, t in zip(attrs, types):
        if t is int:
            t = np.int64
        if t is str:
            t = object
        if attr not in self.expandos:
            self.expandos[attr] = np.zeros(target_size, dtype=t)
            continue
        prev_expando = self.expandos[attr]
        if not np.issubdtype(t, prev_expando.dtype):
            raise TypeError(f"Can't allocate type {t} for attribute {attr}, conflicts with its existing type {prev_expando.dtype}")
        if len(prev_expando) == target_size:
            continue
        prev_count = len(prev_expando)
        self.expandos[attr] = np.zeros(target_size, dtype=prev_expando.dtype)
        self.expandos[attr][:min(prev_count, target_size),] = prev_expando[:min(prev_count, target_size),]