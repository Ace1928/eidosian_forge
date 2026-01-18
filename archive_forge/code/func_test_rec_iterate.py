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
def test_rec_iterate(self):
    descr = np.dtype([('i', int), ('f', float), ('s', '|S3')])
    x = np.rec.array([(1, 1.1, '1.0'), (2, 2.2, '2.0')], dtype=descr)
    x[0].tolist()
    [i for i in x[0]]