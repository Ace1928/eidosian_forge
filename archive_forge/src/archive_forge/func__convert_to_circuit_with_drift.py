from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
def _convert_to_circuit_with_drift(simulator: PhasedFSimEngineSimulator, circuit: cirq.AbstractCircuit) -> cirq.Circuit:

    def map_func(op: cirq.Operation, _) -> cirq.Operation:
        if op.gate is None:
            raise IncompatibleMomentError(f'Operation {op} has a missing gate')
        if isinstance(op.gate, (cirq.MeasurementGate, cirq.WaitGate)) or cirq.num_qubits(op.gate) == 1:
            return op
        translated = simulator.gates_translator(op.gate)
        if translated is None:
            raise IncompatibleMomentError(f'Moment contains non-single qubit operation {op} with unsupported gate')
        a, b = op.qubits
        return simulator.create_gate_with_drift(a, b, translated).on(a, b)
    return cirq.map_operations(circuit, map_func).unfreeze(copy=False)