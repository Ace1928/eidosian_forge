from typing import Dict, Iterable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import Max, Min
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def _translate_min_max_number_of_bins_to_bin_edges(mn: float, mx: float, bins: int, right: bool) -> List[float]:
    """Translates a range and desired number of bins into list of bin edges."""
    rng = (mn, mx)
    mn, mx = (mi + 0.0 for mi in rng)
    if np.isinf(mn) or np.isinf(mx):
        raise ValueError('Cannot specify integer `bins` when input data contains infinity.')
    elif mn == mx:
        mn -= 0.001 * abs(mn) if mn != 0 else 0.001
        mx += 0.001 * abs(mx) if mx != 0 else 0.001
        bins = np.linspace(mn, mx, bins + 1, endpoint=True)
    else:
        bins = np.linspace(mn, mx, bins + 1, endpoint=True)
        adj = (mx - mn) * 0.001
        if right:
            bins[0] -= adj
        else:
            bins[-1] += adj
    return bins