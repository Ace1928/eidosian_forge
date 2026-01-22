import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
class PandasAdapter:
    container_lib = 'pandas'

    def create_container(self, X_output, X_original, columns, inplace=True):
        pd = check_library_installed('pandas')
        columns = get_columns(columns)
        if not inplace or not isinstance(X_output, pd.DataFrame):
            if isinstance(X_output, pd.DataFrame):
                index = X_output.index
            elif isinstance(X_original, pd.DataFrame):
                index = X_original.index
            else:
                index = None
            X_output = pd.DataFrame(X_output, index=index, copy=not inplace)
        if columns is not None:
            return self.rename_columns(X_output, columns)
        return X_output

    def is_supported_container(self, X):
        pd = check_library_installed('pandas')
        return isinstance(X, pd.DataFrame)

    def rename_columns(self, X, columns):
        X.columns = columns
        return X

    def hstack(self, Xs):
        pd = check_library_installed('pandas')
        return pd.concat(Xs, axis=1)