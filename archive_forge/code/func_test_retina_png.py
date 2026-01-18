import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
def test_retina_png():
    here = os.path.dirname(__file__)
    img = display.Image(os.path.join(here, '2x2.png'), retina=True)
    assert img.height == 1
    assert img.width == 1
    data, md = img._repr_png_()
    assert md['width'] == 1
    assert md['height'] == 1