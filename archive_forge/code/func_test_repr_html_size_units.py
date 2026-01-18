from __future__ import annotations
import xml.etree.ElementTree
import pytest
import dask.array as da
from dask.array.svg import draw_sizes
def test_repr_html_size_units():
    pytest.importorskip('jinja2')
    x = da.ones((10000, 5000))
    x = da.ones((3000, 10000), chunks=(1000, 1000))
    text = x._repr_html_()
    assert 'MB' in text or 'MiB' in text
    assert str(x.shape) in text
    assert str(x.dtype) in text
    parses(text)
    x = da.ones((3000, 10000, 50), chunks=(1000, 1000, 10))
    parses(x._repr_html_())