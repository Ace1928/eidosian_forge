from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_summarize_attrs_with_unsafe_attr_name_and_value() -> None:
    attrs = {'<x>': 3, 'y': '<pd.DataFrame>'}
    formatted = fh.summarize_attrs(attrs)
    assert '<dt><span>&lt;x&gt; :</span></dt>' in formatted
    assert '<dt><span>y :</span></dt>' in formatted
    assert '<dd>3</dd>' in formatted
    assert '<dd>&lt;pd.DataFrame&gt;</dd>' in formatted