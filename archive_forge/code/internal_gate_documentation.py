from typing import Any, Dict
from cirq import ops, value
Instatiates an InternalGate.

        Arguments:
            gate_name: Gate class name.
            gate_module: The module of the gate.
            num_qubits: Number of qubits that the gate acts on.
            **kwargs: The named arguments to be passed to the gate constructor.
        