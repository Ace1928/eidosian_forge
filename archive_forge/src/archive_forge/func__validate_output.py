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
def _validate_output(self, result):
    """
        Ensure that the output of each transformer is 2D. Otherwise
        hstack can raise an error or produce incorrect results.
        """
    names = [name for name, _, _, _ in self._iter(fitted=True, column_as_labels=False, skip_drop=True, skip_empty_columns=True)]
    for Xs, name in zip(result, names):
        if not getattr(Xs, 'ndim', 0) == 2 and (not hasattr(Xs, '__dataframe__')):
            raise ValueError("The output of the '{0}' transformer should be 2D (numpy array, scipy sparse array, dataframe).".format(name))
    if _get_output_config('transform', self)['dense'] == 'pandas':
        return
    try:
        import pandas as pd
    except ImportError:
        return
    for Xs, name in zip(result, names):
        if not _is_pandas_df(Xs):
            continue
        for col_name, dtype in Xs.dtypes.to_dict().items():
            if getattr(dtype, 'na_value', None) is not pd.NA:
                continue
            if pd.NA not in Xs[col_name].values:
                continue
            class_name = self.__class__.__name__
            warnings.warn(f"The output of the '{name}' transformer for column '{col_name}' has dtype {dtype} and uses pandas.NA to represent null values. Storing this output in a numpy array can cause errors in downstream scikit-learn estimators, and inefficiencies. Starting with scikit-learn version 1.6, this will raise a ValueError. To avoid this problem you can (i) store the output in a pandas DataFrame by using {class_name}.set_output(transform='pandas') or (ii) modify the input data or the '{name}' transformer to avoid the presence of pandas.NA (for example by using pandas.DataFrame.astype).", FutureWarning)