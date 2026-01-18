import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_param_names():
    model = create_model('tmp')
    assert model.param_names == tuple()
    model.set_param('param1', None)
    assert model.param_names == ('param1',)
    model.set_param('param2', None)
    assert model.param_names == ('param1', 'param2')