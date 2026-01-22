from typing import (
import numpy as np
from cirq import protocols, qis, value
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
Initializes the controlled operation.

        Args:
            controls: The qubits that control the sub-operation.
            sub_operation: The operation that will be controlled.
            control_values: Which control qubit values to apply the sub
                operation.  Either an object that inherits from AbstractControlValues
                or a sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                qubit value (or set of possible values) where that control is
                enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.

        Raises:
            ValueError: If the `control_values` or `control_qid_shape` does not
                match the number of qubits, if the `control_values` are out of
                bounds, if the qubits overlap, or if the sub_operation is not a
                unitary or mixture.
        