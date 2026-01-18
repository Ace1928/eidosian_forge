import inspect
import sys
from IPython.testing import decorators as dec
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import dedent
@dec.skip_win32
def test_win32():
    assert sys.platform != 'win32', "This test can't run under windows"