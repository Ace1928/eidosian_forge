import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator
def _evaluate_linear(self, indices, norm_distances):
    vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))
    shift_norm_distances = [1 - yi for yi in norm_distances]
    shift_indices = [i + 1 for i in indices]
    zipped1 = zip(indices, shift_norm_distances)
    zipped2 = zip(shift_indices, norm_distances)
    hypercube = itertools.product(*zip(zipped1, zipped2))
    value = cp.array([0.0])
    for h in hypercube:
        edge_indices, weights = zip(*h)
        term = cp.asarray(self.values[edge_indices])
        for w in weights:
            term *= w[vslice]
        value = value + term
    return value