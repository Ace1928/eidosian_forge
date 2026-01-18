from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_repr_text_fallback(dataset: xr.Dataset) -> None:
    formatted = fh.dataset_repr(dataset)
    assert "<pre class='xr-text-repr-fallback'>" in formatted