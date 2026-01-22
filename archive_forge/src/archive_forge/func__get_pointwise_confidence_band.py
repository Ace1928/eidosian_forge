from typing import Any, Callable, Optional, Tuple
import warnings
import numpy as np
from scipy.stats import uniform, binom
def _get_pointwise_confidence_band(prob: float, ndraws: int, cdf_at_eval_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the `prob`-level pointwise confidence band."""
    count_lower, count_upper = binom.interval(prob, ndraws, cdf_at_eval_points)
    prob_lower = count_lower / ndraws
    prob_upper = count_upper / ndraws
    return (prob_lower, prob_upper)