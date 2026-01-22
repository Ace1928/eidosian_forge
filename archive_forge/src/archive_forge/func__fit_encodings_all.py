from numbers import Integral, Real
import numpy as np
from ..base import OneToOneFeatureMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import type_of_target
from ..utils.validation import (
from ._encoders import _BaseEncoder
from ._target_encoder_fast import _fit_encoding_fast, _fit_encoding_fast_auto_smooth
def _fit_encodings_all(self, X, y):
    """Fit a target encoding with all the data."""
    from ..preprocessing import LabelBinarizer, LabelEncoder
    check_consistent_length(X, y)
    self._fit(X, handle_unknown='ignore', force_all_finite='allow-nan')
    if self.target_type == 'auto':
        accepted_target_types = ('binary', 'multiclass', 'continuous')
        inferred_type_of_target = type_of_target(y, input_name='y')
        if inferred_type_of_target not in accepted_target_types:
            raise ValueError(f'Unknown label type: Target type was inferred to be {inferred_type_of_target!r}. Only {accepted_target_types} are supported.')
        self.target_type_ = inferred_type_of_target
    else:
        self.target_type_ = self.target_type
    self.classes_ = None
    if self.target_type_ == 'binary':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
    elif self.target_type_ == 'multiclass':
        label_binarizer = LabelBinarizer()
        y = label_binarizer.fit_transform(y)
        self.classes_ = label_binarizer.classes_
    else:
        y = _check_y(y, y_numeric=True, estimator=self)
    self.target_mean_ = np.mean(y, axis=0)
    X_ordinal, X_known_mask = self._transform(X, handle_unknown='ignore', force_all_finite='allow-nan')
    n_categories = np.fromiter((len(category_for_feature) for category_for_feature in self.categories_), dtype=np.int64, count=len(self.categories_))
    if self.target_type_ == 'multiclass':
        encodings = self._fit_encoding_multiclass(X_ordinal, y, n_categories, self.target_mean_)
    else:
        encodings = self._fit_encoding_binary_or_continuous(X_ordinal, y, n_categories, self.target_mean_)
    self.encodings_ = encodings
    return (X_ordinal, X_known_mask, y, n_categories)