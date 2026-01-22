from itertools import groupby
from warnings import warn
import numpy as np
from scipy.sparse import find, coo_matrix
def _call_single(self, t):
    ind = np.searchsorted(self.ts_sorted, t, side=self.side)
    segment = min(max(ind - 1, 0), self.n_segments - 1)
    if not self.ascending:
        segment = self.n_segments - 1 - segment
    return self.interpolants[segment](t)