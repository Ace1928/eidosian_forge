import abc
import fractions
import math
import numbers
from typing import (
import numpy as np
import sympy
from cirq import value, protocols
from cirq.linalg import tolerance
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType
def _approximate_common_period(periods: List[float], approx_denom: int=60, reject_atol: float=1e-08) -> Optional[float]:
    """Finds a value that is nearly an integer multiple of multiple periods.

    The returned value should be the smallest non-negative number with this
    property. If `approx_denom` is too small the computation can fail to satisfy
    the `reject_atol` criteria and return `None`. This is actually desirable
    behavior, since otherwise the code would e.g. return a nonsense value when
    asked to compute the common period of `np.e` and `np.pi`.

    Args:
        periods: The result must be an approximate integer multiple of each of
            these.
        approx_denom: Determines how the floating point values are rounded into
            rational values (so that integer methods such as lcm can be used).
            Each floating point value f_k will be rounded to a rational number
            of the form n_k / approx_denom. If you want to recognize rational
            periods of the form i/d then d should divide `approx_denom`.
        reject_atol: If the computed approximate common period is at least this
            far from an integer multiple of any of the given periods, then it
            is discarded and `None` is returned instead.

    Returns:
        The approximate common period, or else `None` if the given
        `approx_denom` wasn't sufficient to approximate the common period to
        within the given `reject_atol`.
    """
    if not periods:
        return None
    if any((e == 0 for e in periods)):
        return None
    if len(periods) == 1:
        return abs(periods[0])
    approx_rational_periods = [fractions.Fraction(int(np.round(abs(p) * approx_denom)), approx_denom) for p in periods]
    common = float(_common_rational_period(approx_rational_periods))
    for p in periods:
        if p != 0 and abs(p * np.round(common / p) - common) > reject_atol:
            return None
    return common