from typing import Any, Callable, Optional, Tuple
import warnings
import numpy as np
from scipy.stats import uniform, binom
def _simulate_simultaneous_ecdf_band_probability(ndraws: int, eval_points: np.ndarray, cdf_at_eval_points: np.ndarray, prob: float=0.95, rvs: Optional[Callable[[int, Optional[Any]], np.ndarray]]=None, num_trials: int=1000, random_state: Optional[Any]=None) -> float:
    """Estimate probability for simultaneous confidence band using simulation.

    This function simulates the pointwise probability needed to construct pointwise
    confidence bands that form a `prob`-level confidence envelope for the ECDF
    of a sample.
    """
    if rvs is None:
        warnings.warn('Assuming variable is continuous for calibration of pointwise bands. If the variable is discrete, specify random variable sampler `rvs`.', UserWarning)
        rvs = uniform(0, 1).rvs
        eval_points_sim = cdf_at_eval_points
    else:
        eval_points_sim = eval_points
    probs_pointwise = np.empty(num_trials)
    for i in range(num_trials):
        ecdf_at_eval_points = _simulate_ecdf(ndraws, eval_points_sim, rvs, random_state=random_state)
        prob_pointwise = _fit_pointwise_band_probability(ndraws, ecdf_at_eval_points, cdf_at_eval_points)
        probs_pointwise[i] = prob_pointwise
    return np.quantile(probs_pointwise, prob)