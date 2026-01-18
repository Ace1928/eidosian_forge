import warnings
from sympy.testing.pytest import (raises, warns, ignore_warnings,
from sympy.utilities.exceptions import sympy_deprecation_warning
def test_second_argument_should_be_callable_or_string():
    raises(TypeError, lambda: raises('irrelevant', 42))