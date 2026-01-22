from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def append_m(part: pd.DataFrame, name: str, is_valid: bool) -> None:
    nonlocal n_features
    if name == alias.data or name in part.columns:
        if name == alias.data and feature_cols is not None and (part[feature_cols].shape[0] > 0):
            array: Optional[np.ndarray] = part[feature_cols]
        elif part[name].shape[0] > 0:
            array = part[name]
            if name == alias.data:
                array = stack_series(array)
        else:
            array = None
        if name == alias.data and array is not None:
            if n_features == 0:
                n_features = array.shape[1]
            assert n_features == array.shape[1]
        if array is None:
            return
        if is_valid:
            valid_data[name].append(array)
        else:
            train_data[name].append(array)