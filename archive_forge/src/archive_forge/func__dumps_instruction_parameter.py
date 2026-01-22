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
def _dumps_instruction_parameter(param, index_map, use_symengine):
    if isinstance(param, QuantumCircuit):
        type_key = type_keys.Program.CIRCUIT
        data_bytes = common.data_to_binary(param, write_circuit)
    elif isinstance(param, Modifier):
        type_key = type_keys.Value.MODIFIER
        data_bytes = common.data_to_binary(param, _write_modifier)
    elif isinstance(param, range):
        type_key = type_keys.Container.RANGE
        data_bytes = struct.pack(formats.RANGE_PACK, param.start, param.stop, param.step)
    elif isinstance(param, tuple):
        type_key = type_keys.Container.TUPLE
        data_bytes = common.sequence_to_binary(param, _dumps_instruction_parameter, index_map=index_map, use_symengine=use_symengine)
    elif isinstance(param, int):
        type_key = type_keys.Value.INTEGER
        data_bytes = struct.pack('<q', param)
    elif isinstance(param, float):
        type_key = type_keys.Value.FLOAT
        data_bytes = struct.pack('<d', param)
    elif isinstance(param, (Clbit, ClassicalRegister)):
        type_key = type_keys.Value.REGISTER
        data_bytes = _dumps_register(param, index_map)
    else:
        type_key, data_bytes = value.dumps_value(param, index_map=index_map, use_symengine=use_symengine)
    return (type_key, data_bytes)