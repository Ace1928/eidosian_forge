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
def add_vector(self, key, vector):
    """Add one new vector at the given key, into existing slot if available.

        Warning: using this repeatedly is inefficient, requiring a full reallocation & copy,
        if this instance hasn't been preallocated to be ready for such incremental additions.

        Parameters
        ----------

        key: str
            Key identifier of the added vector.
        vector: numpy.ndarray
            1D numpy array with the vector values.

        Returns
        -------
        int
            Index of the newly added vector, so that ``self.vectors[result] == vector`` and
            ``self.index_to_key[result] == key``.

        """
    target_index = self.next_index
    if target_index >= len(self) or self.index_to_key[target_index] is not None:
        target_index = len(self)
        warnings.warn('Adding single vectors to a KeyedVectors which grows by one each time can be costly. Consider adding in batches or preallocating to the required size.', UserWarning)
        self.add_vectors([key], [vector])
        self.allocate_vecattrs()
        self.next_index = target_index + 1
    else:
        self.index_to_key[target_index] = key
        self.key_to_index[key] = target_index
        self.vectors[target_index] = vector
        self.next_index += 1
    return target_index