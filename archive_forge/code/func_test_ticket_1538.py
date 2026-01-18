import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_ticket_1538(self):
    x = np.finfo(np.float32)
    for name in 'eps epsneg max min resolution tiny'.split():
        assert_equal(type(getattr(x, name)), np.float32, err_msg=name)