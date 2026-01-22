import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
class DummyArray(ExtensionArray):

    def __init__(self, data, dtype) -> None:
        self.data = data
        self._dtype = dtype

    def __array__(self, dtype):
        return self.data

    @property
    def dtype(self):
        return self._dtype

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        pass

    def copy(self):
        return self