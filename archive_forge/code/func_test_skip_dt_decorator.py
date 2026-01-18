import inspect
import sys
from IPython.testing import decorators as dec
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import dedent
def test_skip_dt_decorator():
    """Doctest-skipping decorator should preserve the docstring.
    """
    check = 'A function whose doctest we need to skip.\n\n    >>> 1+1\n    3\n    '
    val = doctest_bad.__doc__
    assert dedent(check) == dedent(val), "doctest_bad docstrings don't match"