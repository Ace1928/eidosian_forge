from collections import defaultdict
from typing import Optional, Union
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.transforms import pad
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.lowering import lower_gates
from qiskit.providers import BackendV1, BackendV2
def _meas_start_time():
    _qubit_time_available = defaultdict(int)
    for instruction in scheduled_circuit.data:
        if isinstance(instruction.operation, Measure):
            return _qubit_time_available[instruction.qubits[0]]
        for q in instruction.qubits:
            _qubit_time_available[q] += instruction.operation.duration
    return None