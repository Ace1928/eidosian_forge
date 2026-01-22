import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
class PolarsAdapter:
    container_lib = 'polars'

    def create_container(self, X_output, X_original, columns, inplace=True):
        pl = check_library_installed('polars')
        columns = get_columns(columns)
        columns = columns.tolist() if isinstance(columns, np.ndarray) else columns
        if not inplace or not isinstance(X_output, pl.DataFrame):
            return pl.DataFrame(X_output, schema=columns, orient='row')
        if columns is not None:
            return self.rename_columns(X_output, columns)
        return X_output

    def is_supported_container(self, X):
        pl = check_library_installed('polars')
        return isinstance(X, pl.DataFrame)

    def rename_columns(self, X, columns):
        X.columns = columns
        return X

    def hstack(self, Xs):
        pl = check_library_installed('polars')
        return pl.concat(Xs, how='horizontal')