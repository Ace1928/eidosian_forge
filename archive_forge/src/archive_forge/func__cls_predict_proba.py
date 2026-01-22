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
def _cls_predict_proba(n_classes: int, prediction: PredtT, vstack: Callable) -> PredtT:
    assert len(prediction.shape) <= 2
    if len(prediction.shape) == 2 and prediction.shape[1] == n_classes:
        return prediction
    if len(prediction.shape) == 2 and n_classes == 2 and (prediction.shape[1] >= n_classes):
        return prediction
    classone_probs = prediction
    classzero_probs = 1.0 - classone_probs
    return vstack((classzero_probs, classone_probs)).transpose()