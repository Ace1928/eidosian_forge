from typing import Any, Callable, Optional, Tuple
import warnings
import numpy as np
from scipy.stats import uniform, binom
def _simulate_ecdf(ndraws: int, eval_points: np.ndarray, rvs: Callable[[int, Optional[Any]], np.ndarray], random_state: Optional[Any]=None) -> np.ndarray:
    """Simulate ECDF at the `eval_points` using the given random variable sampler"""
    sample = rvs(ndraws, random_state=random_state)
    sample.sort()
    return compute_ecdf(sample, eval_points)