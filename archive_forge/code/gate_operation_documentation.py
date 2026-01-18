import re
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.ops import raw_types, gate_features, control_values as cv
from cirq.type_workarounds import NotImplementedType
Raise gate to a power, then reapply to the same qubits.

        Only works if the gate implements cirq.ExtrapolatableEffect.
        For extrapolatable gate G this means the following two are equivalent:

            (G ** 1.5)(qubit)  or  G(qubit) ** 1.5

        Args:
            exponent: The amount to scale the gate's effect by.

        Returns:
            A new operation on the same qubits with the scaled gate.
        