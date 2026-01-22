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
def _parse_custom_operation(custom_operations, gate_name, params, version, vectors, registers, use_symengine):
    if version >= 5:
        type_str, num_qubits, num_clbits, definition, num_ctrl_qubits, ctrl_state, base_gate_raw = custom_operations[gate_name]
    else:
        type_str, num_qubits, num_clbits, definition = custom_operations[gate_name]
    if version >= 11:
        gate_name = '_'.join(gate_name.split('_')[:-1])
    type_key = type_keys.CircuitInstruction(type_str)
    if type_key == type_keys.CircuitInstruction.INSTRUCTION:
        inst_obj = Instruction(gate_name, num_qubits, num_clbits, params)
        if definition is not None:
            inst_obj.definition = definition
        return inst_obj
    if type_key == type_keys.CircuitInstruction.GATE:
        inst_obj = Gate(gate_name, num_qubits, params)
        inst_obj.definition = definition
        return inst_obj
    if version >= 5 and type_key == type_keys.CircuitInstruction.CONTROLLED_GATE:
        with io.BytesIO(base_gate_raw) as base_gate_obj:
            base_gate = _read_instruction(base_gate_obj, None, registers, custom_operations, version, vectors, use_symengine)
        if ctrl_state < 2 ** num_ctrl_qubits - 1:
            gate_name = gate_name.rsplit('_', 1)[0]
        inst_obj = ControlledGate(gate_name, num_qubits, params, num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state, base_gate=base_gate)
        inst_obj.definition = definition
        return inst_obj
    if version >= 11 and type_key == type_keys.CircuitInstruction.ANNOTATED_OPERATION:
        with io.BytesIO(base_gate_raw) as base_gate_obj:
            base_gate = _read_instruction(base_gate_obj, None, registers, custom_operations, version, vectors, use_symengine)
        inst_obj = AnnotatedOperation(base_op=base_gate, modifiers=params)
        return inst_obj
    if type_key == type_keys.CircuitInstruction.PAULI_EVOL_GATE:
        return definition
    raise ValueError("Invalid custom instruction type '%s'" % type_str)