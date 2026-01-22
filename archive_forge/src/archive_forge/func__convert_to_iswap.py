from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _convert_to_iswap(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.ISwapPowGate]:
    if isinstance(g, cirq.ISwapPowGate):
        return g
    if isinstance(g, cirq.PhasedISwapPowGate):
        return g._iswap if self._approx_eq_or_symbol(g.phase_exponent, 0) else None
    fsim = self._convert_to_fsim(g)
    return None if fsim is None or not self._approx_eq_or_symbol(fsim.phi, 0) else cirq.ISwapPowGate(exponent=-2 * fsim.theta / np.pi)