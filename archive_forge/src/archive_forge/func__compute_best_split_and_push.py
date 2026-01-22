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
def _compute_best_split_and_push(self, node):
    """Compute the best possible split (SplitInfo) of a given node.

        Also push it in the heap of splittable nodes if gain isn't zero.
        The gain of a node is 0 if either all the leaves are pure
        (best gain = 0), or if no split would satisfy the constraints,
        (min_hessians_to_split, min_gain_to_split, min_samples_leaf)
        """
    node.split_info = self.splitter.find_node_split(n_samples=node.n_samples, histograms=node.histograms, sum_gradients=node.sum_gradients, sum_hessians=node.sum_hessians, value=node.value, lower_bound=node.children_lower_bound, upper_bound=node.children_upper_bound, allowed_features=node.allowed_features)
    if node.split_info.gain <= 0:
        self._finalize_leaf(node)
    else:
        heappush(self.splittable_nodes, node)