import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from scipy.special import softmax
from ._typing import ArrayLike, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import SKLEARN_INSTALLED, XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .config import config_context
from .core import (
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array, _is_pandas_df
from .training import train
def _get_qid(X: ArrayLike, qid: Optional[ArrayLike]) -> Tuple[ArrayLike, Optional[ArrayLike]]:
    """Get the special qid column from X if exists."""
    if (_is_pandas_df(X) or _is_cudf_df(X)) and hasattr(X, 'qid'):
        if qid is not None:
            raise ValueError('Found both the special column `qid` in `X` and the `qid` from the`fit` method. Please remove one of them.')
        q_x = X.qid
        X = X.drop('qid', axis=1)
        return (X, q_x)
    return (X, qid)