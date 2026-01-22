from typing import Any, Union, Iterable
from fractions import Fraction
from decimal import Decimal
import numbers
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
def approx_eq(val: Any, other: Any, *, atol: Union[int, float]=1e-08) -> bool:
    """Approximately compares two objects.

    If `val` implements SupportsApproxEquality protocol then it is invoked and
    takes precedence over all other checks:
     - For primitive numeric types `int` and `float` approximate equality is
       delegated to math.isclose().
     - For complex primitive type the real and imaginary parts are treated
       independently and compared using math.isclose().
     - For `val` and `other` both iterable of the same length, consecutive
       elements are compared recursively. Types of `val` and `other` does not
       necessarily needs to match each other. They just need to be iterable and
       have the same structure.

    Args:
        val: Source object for approximate comparison.
        other: Target object for approximate comparison.
        atol: The minimum absolute tolerance. See np.isclose() documentation for
              details. Defaults to 1e-8 which matches np.isclose() default
              absolute tolerance.

    Returns:
        True if objects are approximately equal, False otherwise.

    Raises:
        AttributeError: If there is insufficient information to determine whether
            the objects are approximately equal.
    """
    approx_eq_getter = getattr(val, '_approx_eq_', None)
    if approx_eq_getter is not None:
        result = approx_eq_getter(other, atol)
        if result is not NotImplemented:
            return result
    other_approx_eq_getter = getattr(other, '_approx_eq_', None)
    if other_approx_eq_getter is not None:
        result = other_approx_eq_getter(val, atol)
        if result is not NotImplemented:
            return result
    if isinstance(val, numbers.Number):
        if not isinstance(other, numbers.Number):
            return False
        result = _isclose(val, other, atol=atol)
        if result is not NotImplemented:
            return result
    if isinstance(val, str):
        return val == other
    if isinstance(val, sympy.Basic) or isinstance(other, sympy.Basic):
        delta = sympy.Abs(other - val).simplify()
        if not delta.is_number:
            raise AttributeError(f'Insufficient information to decide whether expressions are approximately equal [{val}] vs [{other}]')
        return sympy.LessThan(delta, atol) == sympy.true
    if isinstance(val, Iterable) and isinstance(other, Iterable):
        return _approx_eq_iterables(val, other, atol=atol)
    return val == other