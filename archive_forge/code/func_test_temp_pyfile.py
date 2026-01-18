import os
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
def test_temp_pyfile():
    src = 'pass\n'
    fname = tt.temp_pyfile(src)
    assert os.path.isfile(fname)
    with open(fname, encoding='utf-8') as fh2:
        src2 = fh2.read()
    assert src2 == src