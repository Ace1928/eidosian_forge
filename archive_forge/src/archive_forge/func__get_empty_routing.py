import warnings
from collections import Counter
from itertools import chain
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import TransformerMixin, _fit_context, clone
from ..pipeline import _fit_transform_one, _name_estimators, _transform_one
from ..preprocessing import FunctionTransformer
from ..utils import Bunch, _get_column_indices, _safe_indexing
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._metadata_requests import METHODS
from ..utils._param_validation import HasMethods, Hidden, Interval, StrOptions
from ..utils._set_output import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _BaseComposition
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
def _get_empty_routing(self):
    """Return empty routing.

        Used while routing can be disabled.

        TODO: Remove when ``set_config(enable_metadata_routing=False)`` is no
        more an option.
        """
    return Bunch(**{name: Bunch(**{method: {} for method in METHODS}) for name, step, _, _ in self._iter(fitted=False, column_as_labels=False, skip_drop=True, skip_empty_columns=True)})