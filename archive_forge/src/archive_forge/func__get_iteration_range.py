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
def _get_iteration_range(self, iteration_range: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if iteration_range is None or iteration_range[1] == 0:
        try:
            iteration_range = (0, self.best_iteration + 1)
        except AttributeError:
            iteration_range = (0, 0)
    if self.booster == 'gblinear':
        iteration_range = (0, 0)
    return iteration_range