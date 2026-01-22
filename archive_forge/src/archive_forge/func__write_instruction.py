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
def _write_instruction(file_obj, instruction, custom_operations, index_map, use_symengine, version):
    if isinstance(instruction.operation, Instruction):
        gate_class_name = instruction.operation.base_class.__name__
    else:
        gate_class_name = instruction.operation.__class__.__name__
    custom_operations_list = []
    if not hasattr(library, gate_class_name) and (not hasattr(circuit_mod, gate_class_name)) and (not hasattr(controlflow, gate_class_name)) and (gate_class_name != 'Clifford') or gate_class_name == 'Gate' or gate_class_name == 'Instruction' or isinstance(instruction.operation, library.BlueprintCircuit):
        gate_class_name = instruction.operation.name
        if version >= 11:
            gate_class_name = f'{gate_class_name}_{uuid.uuid4().hex}'
        elif instruction.operation.name in {'ucrx_dg', 'ucry_dg', 'ucrz_dg'}:
            gate_class_name = f'{gate_class_name}_{uuid.uuid4()}'
        custom_operations[gate_class_name] = instruction.operation
        custom_operations_list.append(gate_class_name)
    elif gate_class_name in {'ControlledGate', 'AnnotatedOperation'}:
        gate_class_name = instruction.operation.name + '_' + str(uuid.uuid4())
        custom_operations[gate_class_name] = instruction.operation
        custom_operations_list.append(gate_class_name)
    elif isinstance(instruction.operation, library.PauliEvolutionGate):
        gate_class_name = '###PauliEvolutionGate_' + str(uuid.uuid4())
        custom_operations[gate_class_name] = instruction.operation
        custom_operations_list.append(gate_class_name)
    condition_type = type_keys.Condition.NONE
    condition_register = b''
    condition_value = 0
    if (op_condition := getattr(instruction.operation, 'condition', None)) is not None:
        if isinstance(op_condition, expr.Expr):
            condition_type = type_keys.Condition.EXPRESSION
        else:
            condition_type = type_keys.Condition.TWO_TUPLE
            condition_register = _dumps_register(instruction.operation.condition[0], index_map)
            condition_value = int(instruction.operation.condition[1])
    gate_class_name = gate_class_name.encode(common.ENCODE)
    label = getattr(instruction.operation, 'label', None)
    if label:
        label_raw = label.encode(common.ENCODE)
    else:
        label_raw = b''
    if isinstance(instruction.operation, controlflow.SwitchCaseOp):
        instruction_params = [instruction.operation.target, tuple(instruction.operation.cases_specifier())]
    elif isinstance(instruction.operation, Clifford):
        instruction_params = [instruction.operation.tableau]
    elif isinstance(instruction.operation, AnnotatedOperation):
        instruction_params = instruction.operation.modifiers
    else:
        instruction_params = getattr(instruction.operation, 'params', [])
    num_ctrl_qubits = getattr(instruction.operation, 'num_ctrl_qubits', 0)
    ctrl_state = getattr(instruction.operation, 'ctrl_state', 0)
    instruction_raw = struct.pack(formats.CIRCUIT_INSTRUCTION_V2_PACK, len(gate_class_name), len(label_raw), len(instruction_params), instruction.operation.num_qubits, instruction.operation.num_clbits, condition_type.value, len(condition_register), condition_value, num_ctrl_qubits, ctrl_state)
    file_obj.write(instruction_raw)
    file_obj.write(gate_class_name)
    file_obj.write(label_raw)
    if condition_type is type_keys.Condition.EXPRESSION:
        value.write_value(file_obj, op_condition, index_map=index_map)
    else:
        file_obj.write(condition_register)
    for qbit in instruction.qubits:
        instruction_arg_raw = struct.pack(formats.CIRCUIT_INSTRUCTION_ARG_PACK, b'q', index_map['q'][qbit])
        file_obj.write(instruction_arg_raw)
    for clbit in instruction.clbits:
        instruction_arg_raw = struct.pack(formats.CIRCUIT_INSTRUCTION_ARG_PACK, b'c', index_map['c'][clbit])
        file_obj.write(instruction_arg_raw)
    for param in instruction_params:
        type_key, data_bytes = _dumps_instruction_parameter(param, index_map, use_symengine)
        common.write_generic_typed_data(file_obj, type_key, data_bytes)
    return custom_operations_list