import warnings
import numpy as np
import xarray as xr
class MissingMonkeypatchError(Exception):
    """Error specific for the linalg module non-default yet accepted monkeypatch."""