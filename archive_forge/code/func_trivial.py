import inspect
import sys
from IPython.testing import decorators as dec
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import dedent
@dec.as_unittest
def trivial():
    """A trivial test"""
    pass