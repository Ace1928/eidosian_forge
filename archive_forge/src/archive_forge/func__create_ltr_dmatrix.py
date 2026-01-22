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
def _create_ltr_dmatrix(self, ref: Optional[DMatrix], data: ArrayLike, qid: ArrayLike, **kwargs: Any) -> DMatrix:
    data, qid = _get_qid(data, qid)
    if kwargs.get('group', None) is None and qid is None:
        raise ValueError('Either `group` or `qid` is required for ranking task')
    return super()._create_dmatrix(ref=ref, data=data, qid=qid, **kwargs)