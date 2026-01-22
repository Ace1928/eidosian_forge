from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _get_const_index(exog):
    """
    Returns a boolean array of non-constant column indices in exog and
    an scalar array of where the constant is or None
    """
    effects_idx = exog.var(0) != 0
    if np.any(~effects_idx):
        const_idx = np.where(~effects_idx)[0]
    else:
        const_idx = None
    return (effects_idx, const_idx)