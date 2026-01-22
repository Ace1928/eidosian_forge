from typing import Any, Callable, Optional, Tuple
import warnings
import numpy as np
from scipy.stats import uniform, binom
def _fit_pointwise_band_probability(ndraws: int, ecdf_at_eval_points: np.ndarray, cdf_at_eval_points: np.ndarray) -> float:
    """Compute the smallest marginal probability of a pointwise confidence band that
    contains the ECDF."""
    ecdf_scaled = (ndraws * ecdf_at_eval_points).astype(int)
    prob_lower_tail = np.amin(binom.cdf(ecdf_scaled, ndraws, cdf_at_eval_points))
    prob_upper_tail = np.amin(binom.sf(ecdf_scaled - 1, ndraws, cdf_at_eval_points))
    prob_pointwise = 1 - 2 * min(prob_lower_tail, prob_upper_tail)
    return prob_pointwise