from typing import Any, Callable, Optional, overload, TypeVar, TYPE_CHECKING, Union
Returns `val**factor` of the given value, if defined.

    Values define an extrapolation by defining a __pow__(self, exponent) method.
    Note that the method may return NotImplemented to indicate a particular
    extrapolation can't be done.

    Args:
        val: The value or iterable of values to invert.
        exponent: The extrapolation factor. For example, if this is 0.5 and val
            is a gate then the caller is asking for a square root of the gate.
        default: Determines the fallback behavior when `val` doesn't have
            an extrapolation defined. If `default` is not set and that occurs,
            a TypeError is raised instead.

    Returns:
        If `val` has a __pow__ method that returns something besides
        NotImplemented, that result is returned. Otherwise, if a default value
        was specified, the default value is returned.

    Raises:
        TypeError: `val` doesn't have a __pow__ method (or that method returned
            NotImplemented) and no `default` value was specified.
    