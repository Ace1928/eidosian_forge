import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def _handle_NA_raise(self, values, is_NAs, origins):
    for is_NA, origin in zip(is_NAs, origins):
        if np.any(is_NA):
            raise PatsyError('factor contains missing values', origin)
    return values