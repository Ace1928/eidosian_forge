from typing import (
from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types
Validates whether the given `cirq.Operation` is contained in this Gateset.

        The containment checks are handled as follows:

        - For any operation which has an underlying gate (i.e. `op.gate` is not None):
            - Containment is checked via `self.__contains__` which further checks for containment
                in any of the underlying gate families.
        - For all other types of operations (eg: `cirq.CircuitOperation`,
            etc):
            - The behavior is controlled via flags passed to the constructor.

        Users should override this method to define custom behavior for operations that do not
        have an underlying `cirq.Gate`.

        Args:
            op: The `cirq.Operation` instance to check containment for.
        