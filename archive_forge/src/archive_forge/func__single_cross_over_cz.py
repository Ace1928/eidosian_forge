from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _single_cross_over_cz(op: ops.Operation, qubit_with_w: 'cirq.Qid') -> 'cirq.OP_TREE':
    """Crosses exactly one W flip over a partial CZ.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:

        ──────────@─────
                  │
        ───W(a)───@^t───


        ≡ ───@──────O──────@────────────────────
             |      |      │                      (split into on/off cases)
          ───W(a)───W(a)───@^t──────────────────

        ≡ ───@─────────────@─────────────O──────
             |             │             |        (off doesn't interact with on)
          ───W(a)──────────@^t───────────W(a)───

        ≡ ───────────Z^t───@──────@──────O──────
                           │      |      |        (crossing causes kickback)
          ─────────────────@^-t───W(a)───W(a)───  (X Z^t X Z^-t = exp(pi t) I)

        ≡ ───────────Z^t───@────────────────────
                           │                      (merge on/off cases)
          ─────────────────@^-t───W(a)──────────

        ≡ ───Z^t───@──────────────
                   │
          ─────────@^-t───W(a)────
    """
    t = cast(value.TParamVal, _try_get_known_cz_half_turns(op))
    other_qubit = op.qubits[0] if qubit_with_w == op.qubits[1] else op.qubits[1]
    negated_cz = ops.CZ(*op.qubits) ** (-t)
    kickback = ops.Z(other_qubit) ** t
    return [kickback, negated_cz]