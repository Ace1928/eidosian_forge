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
def _common_rational_period(rational_periods: List[fractions.Fraction]) -> fractions.Fraction:
    """Finds the least common integer multiple of some fractions.

    The solution is the smallest positive integer c such that there
    exists integers n_k satisfying p_k * n_k = c for all k.
    """
    assert rational_periods, 'no well-defined solution for an empty list'
    common_denom = _lcm((p.denominator for p in rational_periods))
    int_periods = [p.numerator * common_denom // p.denominator for p in rational_periods]
    int_common_period = _lcm(int_periods)
    return fractions.Fraction(int_common_period, common_denom)