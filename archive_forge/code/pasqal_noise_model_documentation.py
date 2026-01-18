from typing import List, Dict, Sequence, Any
import cirq
import cirq_pasqal
Find the string representation for a given operation.

        Args:
            cirq_op: A cirq operation.

        Returns:
            String representing the gate operations.

        Raises:
            ValueError: If the operations gate is not supported.
        