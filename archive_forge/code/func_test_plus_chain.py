import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_plus_chain():
    with Model.define_operators({'+': lambda a, b: a}):
        m = create_model(name='a') + create_model(name='b') + create_model(name='c') + create_model(name='d')
        assert m.name == 'a'