from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_repr_of_multiindex(multiindex: xr.Dataset) -> None:
    formatted = fh.dataset_repr(multiindex)
    assert '(x)' in formatted