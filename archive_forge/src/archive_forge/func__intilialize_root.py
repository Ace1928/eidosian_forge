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
def _intilialize_root(self, gradients, hessians, hessians_are_constant):
    """Initialize root node and finalize it if needed."""
    n_samples = self.X_binned.shape[0]
    depth = 0
    sum_gradients = sum_parallel(gradients, self.n_threads)
    if self.histogram_builder.hessians_are_constant:
        sum_hessians = hessians[0] * n_samples
    else:
        sum_hessians = sum_parallel(hessians, self.n_threads)
    self.root = TreeNode(depth=depth, sample_indices=self.splitter.partition, sum_gradients=sum_gradients, sum_hessians=sum_hessians, value=0)
    self.root.partition_start = 0
    self.root.partition_stop = n_samples
    if self.root.n_samples < 2 * self.min_samples_leaf:
        self._finalize_leaf(self.root)
        return
    if sum_hessians < self.splitter.min_hessian_to_split:
        self._finalize_leaf(self.root)
        return
    if self.interaction_cst is not None:
        self.root.interaction_cst_indices = range(len(self.interaction_cst))
        allowed_features = set().union(*self.interaction_cst)
        self.root.allowed_features = np.fromiter(allowed_features, dtype=np.uint32, count=len(allowed_features))
    tic = time()
    self.root.histograms = self.histogram_builder.compute_histograms_brute(self.root.sample_indices, self.root.allowed_features)
    self.total_compute_hist_time += time() - tic
    tic = time()
    self._compute_best_split_and_push(self.root)
    self.total_find_split_time += time() - tic