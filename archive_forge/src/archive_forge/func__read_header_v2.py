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
def _read_header_v2(file_obj, version, vectors, metadata_deserializer=None):
    data = formats.CIRCUIT_HEADER_V2._make(struct.unpack(formats.CIRCUIT_HEADER_V2_PACK, file_obj.read(formats.CIRCUIT_HEADER_V2_SIZE)))
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    global_phase = value.loads_value(data.global_phase_type, file_obj.read(data.global_phase_size), version=version, vectors=vectors)
    header = {'global_phase': global_phase, 'num_qubits': data.num_qubits, 'num_clbits': data.num_clbits, 'num_registers': data.num_registers, 'num_instructions': data.num_instructions}
    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    return (header, name, metadata)