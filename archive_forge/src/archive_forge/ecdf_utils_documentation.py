from typing import Any, Callable, Optional, Tuple
import warnings
import numpy as np
from scipy.stats import uniform, binom
Estimate probability for simultaneous confidence band using simulation.

    This function simulates the pointwise probability needed to construct pointwise
    confidence bands that form a `prob`-level confidence envelope for the ECDF
    of a sample.
    