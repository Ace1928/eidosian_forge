from typing import Any, cast, Optional, Type, Union
from cirq.ops import gateset, raw_types, parallel_gate, eigen_gate
from cirq import protocols
Inits ParallelGateFamily

        Args:
            gate: The gate which can act in parallel. It can be a python `type` inheriting from
                `cirq.Gate` or a non-parameterized instance of a `cirq.Gate`. If an instance of
                `cirq.ParallelGate` is passed, then the corresponding `gate.sub_gate` is used.
            name: The name of the gate family.
            description: Human readable description of the gate family.
            max_parallel_allowed: The maximum number of qubits on which a given gate `g`
            can act on. If None, then any number of qubits are allowed.
        