from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
Crosses two W flips over a partial CZ.

    [Where W(a) is shorthand for PhasedX(phase_exponent=a).]

    Uses the following identity:

        ───W(a)───@─────
                  │
        ───W(b)───@^t───


        ≡ ──────────@────────────W(a)───
                    │                     (single-cross top W over CZ)
          ───W(b)───@^-t─────────Z^t────


        ≡ ──────────@─────Z^-t───W(a)───
                    │                     (single-cross bottom W over CZ)
          ──────────@^t───W(b)───Z^t────


        ≡ ──────────@─────W(a)───Z^t────
                    │                     (flip over Z^-t)
          ──────────@^t───W(b)───Z^t────


        ≡ ──────────@─────W(a+t/2)──────
                    │                     (absorb Zs into Ws)
          ──────────@^t───W(b+t/2)──────

        ≡ ───@─────W(a+t/2)───
             │
          ───@^t───W(b+t/2)───
    