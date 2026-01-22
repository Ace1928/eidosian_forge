from __future__ import annotations
from abc import (
from typing import (
from pandas.compat import (
from pandas.core.dtypes.common import is_list_like
class ArrowAccessor(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, data, validation_msg: str) -> None:
        self._data = data
        self._validation_msg = validation_msg
        self._validate(data)

    @abstractmethod
    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        pass

    def _validate(self, data):
        dtype = data.dtype
        if not isinstance(dtype, ArrowDtype):
            raise AttributeError(self._validation_msg.format(dtype=dtype))
        if not self._is_valid_pyarrow_dtype(dtype.pyarrow_dtype):
            raise AttributeError(self._validation_msg.format(dtype=dtype))

    @property
    def _pa_array(self):
        return self._data.array._pa_array