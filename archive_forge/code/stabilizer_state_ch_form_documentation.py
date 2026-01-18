from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
Applies a Z projector on the q'th qubit.

        Returns: a normalized state with Z_q |psi> = z |psi>
        