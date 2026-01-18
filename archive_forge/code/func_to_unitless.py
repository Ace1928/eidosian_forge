from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def to_unitless(value, new_unit=None):
    """Nondimensionalization of a quantity.

    Parameters
    ----------
    value: quantity
    new_unit: unit

    Examples
    --------
    >>> '%.1g' % to_unitless(1*default_units.metre, default_units.nm)
    '1e+09'
    >>> '%.1g %.1g' % tuple(to_unitless([1*default_units.m, 1*default_units.mm], default_units.nm))
    '1e+09 1e+06'

    """
    integer_one = 1
    if new_unit is None:
        new_unit = pq.dimensionless
    if isinstance(value, (list, tuple)):
        return np.array([to_unitless(elem, new_unit) for elem in value])
    elif isinstance(value, np.ndarray) and (not hasattr(value, 'rescale')):
        if is_unitless(new_unit) and new_unit == 1 and (value.dtype != object):
            return value
        return np.array([to_unitless(elem, new_unit) for elem in value])
    elif isinstance(value, dict):
        new_value = dict(value.items())
        for k in value:
            new_value[k] = to_unitless(value[k], new_unit)
        return new_value
    elif isinstance(value, (int, float)) and new_unit is integer_one or new_unit is None:
        return value
    elif isinstance(value, str):
        raise ValueError('str not supported')
    else:
        try:
            try:
                mag = magnitude(value)
                unt = unit_of(value)
                conv = rescale(unt / new_unit, pq.dimensionless)
                result = np.array(mag) * conv
            except AttributeError:
                if new_unit == pq.dimensionless:
                    return value
                else:
                    raise
            else:
                if result.ndim == 0:
                    return float(result)
                else:
                    return np.asarray(result)
        except TypeError:
            return np.array([to_unitless(elem, new_unit) for elem in value])