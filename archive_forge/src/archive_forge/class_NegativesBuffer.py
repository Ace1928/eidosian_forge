import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
class NegativesBuffer:
    """Buffer and return negative samples."""

    def __init__(self, items):
        """Initialize instance from list or numpy array of samples.

        Parameters
        ----------
        items : list/numpy.array
            List or array containing negative samples.

        """
        self._items = items
        self._current_index = 0

    def num_items(self):
        """Get the number of items remaining in the buffer.

        Returns
        -------
        int
            Number of items in the buffer that haven't been consumed yet.

        """
        return len(self._items) - self._current_index

    def get_items(self, num_items):
        """Get the next `num_items` from buffer.

        Parameters
        ----------
        num_items : int
            Number of items to fetch.

        Returns
        -------
        numpy.array or list
            Slice containing `num_items` items from the original data.

        Notes
        -----
        No error is raised if less than `num_items` items are remaining,
        simply all the remaining items are returned.

        """
        start_index = self._current_index
        end_index = start_index + num_items
        self._current_index += num_items
        return self._items[start_index:end_index]