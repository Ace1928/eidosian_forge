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
def validate_or_none(meta: Optional[Sequence], name: str) -> Sequence:
    if meta is None:
        return [None] * n_validation
    if len(meta) != n_validation:
        raise ValueError(f"{name}'s length does not equal `eval_set`'s length, " + f'expecting {n_validation}, got {len(meta)}')
    return meta