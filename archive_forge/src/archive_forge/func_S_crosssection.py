import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def S_crosssection(x, group):
    """inner covariance matrix for White on group sums sandwich

    I guess for a single categorical group only,
    categorical group, can also be the product/intersection of groups

    This is used by cov_cluster and indirectly verified

    """
    x_group_sums = group_sums(x, group).T
    return S_white_simple(x_group_sums)