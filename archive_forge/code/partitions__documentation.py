from mpmath.libmp import (fzero, from_int, from_rational,
from sympy.core.numbers import igcd
from .residue_ntheory import (_sqrt_mod_prime_power,
import math

    Calculate the partition function P(n), i.e. the number of ways that
    n can be written as a sum of positive integers.

    P(n) is computed using the Hardy-Ramanujan-Rademacher formula [1]_.


    The correctness of this implementation has been tested through $10^{10}$.

    Examples
    ========

    >>> from sympy.ntheory import npartitions
    >>> npartitions(25)
    1958

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PartitionFunctionP.html

    