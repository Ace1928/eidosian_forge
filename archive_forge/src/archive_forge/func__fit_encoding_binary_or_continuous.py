from numbers import Integral, Real
import numpy as np
from ..base import OneToOneFeatureMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import type_of_target
from ..utils.validation import (
from ._encoders import _BaseEncoder
from ._target_encoder_fast import _fit_encoding_fast, _fit_encoding_fast_auto_smooth
def _fit_encoding_binary_or_continuous(self, X_ordinal, y, n_categories, target_mean):
    """Learn target encodings."""
    if self.smooth == 'auto':
        y_variance = np.var(y)
        encodings = _fit_encoding_fast_auto_smooth(X_ordinal, y, n_categories, target_mean, y_variance)
    else:
        encodings = _fit_encoding_fast(X_ordinal, y, n_categories, self.smooth, target_mean)
    return encodings