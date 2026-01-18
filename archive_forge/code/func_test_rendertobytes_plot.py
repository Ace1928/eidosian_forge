import pytest
import contextlib
import os
import tempfile
from rpy2.robjects.packages import importr, data
from rpy2.robjects import r
from rpy2.robjects.lib import grdevices
def test_rendertobytes_plot():
    with grdevices.render_to_bytesio(grdevices.png) as b:
        r(' plot(0) ')
    assert len(b.getvalue()) > 0