from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_repr_of_nonstr_dataset(dataset: xr.Dataset) -> None:
    ds = dataset.copy()
    ds.attrs[1] = 'Test value'
    ds[2] = ds['tmin']
    formatted = fh.dataset_repr(ds)
    assert '<dt><span>1 :</span></dt><dd>Test value</dd>' in formatted
    assert "<div class='xr-var-name'><span>2</span>" in formatted