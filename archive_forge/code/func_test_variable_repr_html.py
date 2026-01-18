from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_variable_repr_html() -> None:
    v = xr.Variable(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
    assert hasattr(v, '_repr_html_')
    with xr.set_options(display_style='html'):
        html = v._repr_html_().strip()
    assert html.startswith('<div') and html.endswith('</div>')
    assert 'xarray.Variable' in html