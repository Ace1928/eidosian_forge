from typing import (
import sympy
from cirq import protocols, value
from cirq.ops import op_tree, raw_types
Initializes a `ClassicallyControlledOperation`.

        Multiple consecutive `ClassicallyControlledOperation` layers are
        squashed when possible, so one should not depend on a specific number
        of layers.

        Args:
            sub_operation: The operation to gate with a classical control
                condition.
            conditions: A sequence of measurement keys, or strings that can be
                parsed into measurement keys.

        Raises:
            ValueError: If an unsupported gate is being classically
                controlled.
        