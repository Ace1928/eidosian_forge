from numbers import Integral, Real
import numpy as np
from ..base import OneToOneFeatureMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import type_of_target
from ..utils.validation import (
from ._encoders import _BaseEncoder
from ._target_encoder_fast import _fit_encoding_fast, _fit_encoding_fast_auto_smooth
def _fit_encoding_multiclass(self, X_ordinal, y, n_categories, target_mean):
    """Learn multiclass encodings.

        Learn encodings for each class (c) then reorder encodings such that
        the same features (f) are grouped together. `reorder_index` enables
        converting from:
        f0_c0, f1_c0, f0_c1, f1_c1, f0_c2, f1_c2
        to:
        f0_c0, f0_c1, f0_c2, f1_c0, f1_c1, f1_c2
        """
    n_features = self.n_features_in_
    n_classes = len(self.classes_)
    encodings = []
    for i in range(n_classes):
        y_class = y[:, i]
        encoding = self._fit_encoding_binary_or_continuous(X_ordinal, y_class, n_categories, target_mean[i])
        encodings.extend(encoding)
    reorder_index = (idx for start in range(n_features) for idx in range(start, n_classes * n_features, n_features))
    return [encodings[idx] for idx in reorder_index]