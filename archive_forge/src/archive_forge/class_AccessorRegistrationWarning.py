from __future__ import annotations
import warnings
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""