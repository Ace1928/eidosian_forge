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
def _is_empty_column_selection(column):
    """
    Return True if the column selection is empty (empty list or all-False
    boolean array).

    """
    if hasattr(column, 'dtype') and np.issubdtype(column.dtype, np.bool_):
        return not column.any()
    elif hasattr(column, '__len__'):
        return len(column) == 0 or (all((isinstance(col, bool) for col in column)) and (not any(column)))
    else:
        return False