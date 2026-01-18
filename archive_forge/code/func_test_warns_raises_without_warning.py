import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_warns_raises_without_warning():
    with raises(Failed):
        with warns(UserWarning):
            pass