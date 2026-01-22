from collections import defaultdict
import io
import json
import struct
import uuid
import warnings
import numpy as np
from qiskit import circuit as circuit_mod
from qiskit.circuit import library, controlflow, CircuitInstruction, ControlFlowOp
from qiskit.circuit.classical import expr
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.singleton import SingletonInstruction, SingletonGate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.annotated_operation import (
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.qpy import common, formats, type_keys
from qiskit.qpy.binary_io import value, schedules
from qiskit.quantum_info.operators import SparsePauliOp, Clifford
from qiskit.synthesis import evolution as evo_synth
from qiskit.transpiler.layout import Layout, TranspileLayout
def _write_calibrations(file_obj, calibrations, metadata_serializer):
    flatten_dict = {}
    for gate, caldef in calibrations.items():
        for (qubits, params), schedule in caldef.items():
            key = (gate, qubits, params)
            flatten_dict[key] = schedule
    header = struct.pack(formats.CALIBRATION_PACK, len(flatten_dict))
    file_obj.write(header)
    for (name, qubits, params), schedule in flatten_dict.items():
        name_bytes = name.encode(common.ENCODE)
        defheader = struct.pack(formats.CALIBRATION_DEF_PACK, len(name_bytes), len(qubits), len(params), type_keys.Program.assign(schedule))
        file_obj.write(defheader)
        file_obj.write(name_bytes)
        for qubit in qubits:
            file_obj.write(struct.pack('!q', qubit))
        for param in params:
            value.write_value(file_obj, param)
        schedules.write_schedule_block(file_obj, schedule, metadata_serializer)