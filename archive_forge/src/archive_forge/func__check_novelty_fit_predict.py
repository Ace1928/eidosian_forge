import warnings
from numbers import Real
import numpy as np
from ..base import OutlierMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.metaestimators import available_if
from ..utils.validation import check_is_fitted
from ._base import KNeighborsMixin, NeighborsBase
def _check_novelty_fit_predict(self):
    if self.novelty:
        msg = 'fit_predict is not available when novelty=True. Use novelty=False if you want to predict on the training set.'
        raise AttributeError(msg)
    return True