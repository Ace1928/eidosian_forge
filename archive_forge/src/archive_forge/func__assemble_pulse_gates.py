import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from qiskit.assembler.run_config import RunConfig
from qiskit.assembler.assemble_schedules import _assemble_instructions as _assemble_schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.exceptions import QiskitError
from qiskit.qobj import (
from qiskit.utils.parallel import parallel_map
def _assemble_pulse_gates(circuit: QuantumCircuit, run_config: RunConfig) -> Tuple[Optional[QasmExperimentCalibrations], Optional[PulseLibrary]]:
    """Assemble and return the circuit calibrations and associated pulse library, if there are any.
    The calibrations themselves may reference the pulse library which is returned as a dict.

    Args:
        circuit: circuit which may have pulse calibrations
        run_config: configuration of the runtime environment

    Returns:
        The calibrations and pulse library, if there are any
    """
    if not circuit.calibrations:
        return (None, None)
    if not hasattr(run_config, 'parametric_pulses'):
        run_config.parametric_pulses = []
    calibrations = []
    pulse_library = {}
    for gate, cals in circuit.calibrations.items():
        for (qubits, params), schedule in cals.items():
            qobj_instructions, _ = _assemble_schedule(schedule, converters.InstructionToQobjConverter(PulseQobjInstruction), run_config, pulse_library)
            calibrations.append(GateCalibration(str(gate), list(qubits), list(params), qobj_instructions))
    return (QasmExperimentCalibrations(gates=calibrations), pulse_library)