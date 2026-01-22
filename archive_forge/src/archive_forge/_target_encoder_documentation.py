from numbers import Integral, Real
import numpy as np
from ..base import OneToOneFeatureMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import type_of_target
from ..utils.validation import (
from ._encoders import _BaseEncoder
from ._target_encoder_fast import _fit_encoding_fast, _fit_encoding_fast_auto_smooth
Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names. `feature_names_in_` is used unless it is
            not defined, in which case the following input feature names are
            generated: `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            When `type_of_target_` is "multiclass" the names are of the format
            '<feature_name>_<class_name>'.
        