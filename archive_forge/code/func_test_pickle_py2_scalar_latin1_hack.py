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
def test_pickle_py2_scalar_latin1_hack(self):
    datas = [(np.str_('毒'), b"cnumpy.core.multiarray\nscalar\np0\n(cnumpy\ndtype\np1\n(S'U1'\np2\nI0\nI1\ntp3\nRp4\n(I3\nS'<'\np5\nNNNI4\nI4\nI0\ntp6\nbS'\\xd2k\\x00\\x00'\np7\ntp8\nRp9\n.", 'invalid'), (np.float64(9e+123), b"cnumpy.core.multiarray\nscalar\np0\n(cnumpy\ndtype\np1\n(S'f8'\np2\nI0\nI1\ntp3\nRp4\n(I3\nS'<'\np5\nNNNI-1\nI-1\nI0\ntp6\nbS'O\\x81\\xb7Z\\xaa:\\xabY'\np7\ntp8\nRp9\n.", 'invalid'), (np.bytes_(b'\x9c'), b"cnumpy.core.multiarray\nscalar\np0\n(cnumpy\ndtype\np1\n(S'S1'\np2\nI0\nI1\ntp3\nRp4\n(I3\nS'|'\np5\nNNNI1\nI1\nI0\ntp6\nbS'\\x9c'\np7\ntp8\nRp9\n.", 'different')]
    for original, data, koi8r_validity in datas:
        result = pickle.loads(data, encoding='latin1')
        assert_equal(result, original)
        if koi8r_validity == 'different':
            result = pickle.loads(data, encoding='koi8-r')
            assert_(result != original)
        elif koi8r_validity == 'invalid':
            assert_raises(ValueError, pickle.loads, data, encoding='koi8-r')
        else:
            raise ValueError(koi8r_validity)