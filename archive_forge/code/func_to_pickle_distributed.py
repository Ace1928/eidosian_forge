from __future__ import annotations
import pickle
import warnings
from typing import TYPE_CHECKING, Union
import pandas
from pandas._typing import CompressionOptions, StorageOptions
from pandas.core.dtypes.dtypes import SparseDtype
from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.io import to_dask, to_ray
from modin.utils import _inherit_docstrings
def to_pickle_distributed(self, filepath_or_buffer, compression: CompressionOptions='infer', protocol: int=pickle.HIGHEST_PROTOCOL, storage_options: StorageOptions=None) -> None:
    warnings.warn('`DataFrame.modin.to_pickle_distributed` is deprecated and will be removed in a future version. ' + 'Please use `DataFrame.modin.to_pickle_glob` instead.', category=FutureWarning)
    return self.to_pickle_glob(filepath_or_buffer, compression, protocol, storage_options)