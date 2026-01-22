import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple
import sympy.parsing.sympy_parser as sympy_parser
import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString
def _apply_cnots(prevh: Tuple[int, ...], currh: Tuple[int, ...]):
    cnots: List[Tuple[int, int]] = []
    cnots.extend(((prevh[i], prevh[-1]) for i in range(len(prevh) - 1)))
    cnots.extend(((currh[i], currh[-1]) for i in range(len(currh) - 1)))
    cnots = _simplify_cnots(cnots)
    for gate in (cirq.CNOT(qubits[c], qubits[t]) for c, t in cnots):
        yield gate