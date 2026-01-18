import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_from_names(self):
    ra = np.rec.array([(1, 'abc', 3.700000286102295, 0), (2, 'xy', 6.699999809265137, 1), (0, ' ', 0.4000000059604645, 0)], names='c1, c2, c3, c4')
    pa = np.rec.fromrecords([(1, 'abc', 3.700000286102295, 0), (2, 'xy', 6.699999809265137, 1), (0, ' ', 0.4000000059604645, 0)], names='c1, c2, c3, c4')
    assert_(ra.dtype == pa.dtype)
    assert_(ra.shape == pa.shape)
    for k in range(len(ra)):
        assert_(ra[k].item() == pa[k].item())