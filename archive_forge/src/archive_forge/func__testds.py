import numpy as np
from xarray import DataArray, Dataset, Variable
def _testds(ds: Dataset) -> None:
    assert isinstance(ds, Dataset)