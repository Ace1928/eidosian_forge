from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def _read_csr_matrix_from_unwrapped_spark_vec(part: pd.DataFrame) -> csr_matrix:
    csr_indices_list, csr_indptr_list, csr_values_list = ([], [0], [])
    n_features = 0
    for vec_type, vec_size_, vec_indices, vec_values in zip(part.featureVectorType, part.featureVectorSize, part.featureVectorIndices, part.featureVectorValues):
        if vec_type == 0:
            vec_size = int(vec_size_)
            csr_indices = vec_indices
            csr_values = vec_values
        else:
            vec_size = len(vec_values)
            csr_indices = np.arange(vec_size, dtype=np.int32)
            csr_values = vec_values
        if n_features == 0:
            n_features = vec_size
        assert n_features == vec_size
        csr_indices_list.append(csr_indices)
        csr_indptr_list.append(csr_indptr_list[-1] + len(csr_indices))
        csr_values_list.append(csr_values)
    csr_indptr_arr = np.array(csr_indptr_list)
    csr_indices_arr = np.concatenate(csr_indices_list)
    csr_values_arr = np.concatenate(csr_values_list)
    return csr_matrix((csr_values_arr, csr_indices_arr, csr_indptr_arr), shape=(len(part), n_features))