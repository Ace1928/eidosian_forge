import numbers
from heapq import heappop, heappush
from timeit import default_timer as time
import numpy as np
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from ._bitset import set_raw_bitset_from_binned_bitset
from .common import (
from .histogram import HistogramBuilder
from .predictor import TreePredictor
from .splitting import Splitter
from .utils import sum_parallel
Make a TreePredictor object out of the current tree.

        Parameters
        ----------
        binning_thresholds : array-like of floats
            Corresponds to the bin_thresholds_ attribute of the BinMapper.
            For each feature, this stores:

            - the bin frontiers for continuous features
            - the unique raw category values for categorical features

        Returns
        -------
        A TreePredictor object.
        