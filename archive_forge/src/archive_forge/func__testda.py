import numpy as np
from xarray import DataArray, Dataset, Variable
def _testda(da: DataArray) -> None:
    assert isinstance(da, DataArray)