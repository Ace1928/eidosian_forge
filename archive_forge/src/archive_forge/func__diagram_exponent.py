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
def _diagram_exponent(self, args: 'protocols.CircuitDiagramInfoArgs', *, ignore_global_phase: bool=True):
    """The exponent to use in circuit diagrams.

        Basically, this just canonicalizes the exponent in a way that is
        insensitive to global phase. Only relative phases affect the "true"
        exponent period, and since we omit global phase detail in diagrams this
        is the appropriate canonicalization to use. To use the absolute period
        instead of the relative period (e.g. for when printing Rx(rads) style
        symbols, where rads=pi and rads=-pi are equivalent but should produce
        different text) set 'ignore_global_phase' to False.

        Note that the exponent is canonicalized into the range
            (-period/2, period/2]
        and that this canonicalization happens after rounding, so that e.g.
        X^-0.999999 shows as X instead of X^-1 when using a digit precision of
        3.

        Args:
            args: The diagram args being used to produce the diagram.
            ignore_global_phase: Determines whether the global phase of the
                operation is considered when computing the period of the
                exponent.

        Returns:
            A rounded canonicalized exponent.
        """
    if not isinstance(self._exponent, (int, float)):
        return self._exponent
    result = float(self._exponent)
    if ignore_global_phase:
        shifts = list(self._eigen_shifts())
        relative_shifts = {e - shifts[0] for e in shifts[1:]}
        relative_periods = [abs(2 / e) for e in relative_shifts if e != 0]
        diagram_period = _approximate_common_period(relative_periods)
    else:
        diagram_period = self._period()
    if diagram_period is None:
        return result
    if args.precision is not None:
        result = np.around(result, args.precision).item()
    h = diagram_period / 2
    if not -h < result <= h:
        result = h - result
        result %= diagram_period
        result = h - result
    return result