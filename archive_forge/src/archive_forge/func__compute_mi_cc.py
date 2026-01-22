from numbers import Integral
import numpy as np
from scipy.sparse import issparse
from scipy.special import digamma
from ..metrics.cluster import mutual_info_score
from ..neighbors import KDTree, NearestNeighbors
from ..preprocessing import scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_array, check_X_y
def _compute_mi_cc(x, y, n_neighbors):
    """Compute mutual information between two continuous variables.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,)
        Samples of two continuous random variables, must have an identical
        shape.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information in nat units. If it turned out to be
        negative it is replaced by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    n_samples = x.size
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    xy = np.hstack((x, y))
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)
    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)
    kd = KDTree(x, metric='chebyshev')
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0
    kd = KDTree(y, metric='chebyshev')
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0
    mi = digamma(n_samples) + digamma(n_neighbors) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))
    return max(0, mi)