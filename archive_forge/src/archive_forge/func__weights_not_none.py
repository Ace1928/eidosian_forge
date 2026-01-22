from abc import abstractmethod
from numbers import Integral
import numpy as np
from ..base import (
from ..exceptions import NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._param_validation import StrOptions
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
@property
def _weights_not_none(self):
    """Get the weights of not `None` estimators."""
    if self.weights is None:
        return None
    return [w for est, w in zip(self.estimators, self.weights) if est[1] != 'drop']