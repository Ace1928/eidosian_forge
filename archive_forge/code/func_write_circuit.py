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
def write_circuit(file_obj, circuit, metadata_serializer=None, use_symengine=False, version=common.QPY_VERSION):
    """Write a single QuantumCircuit object in the file like object.

    Args:
        file_obj (FILE): The file like object to write the circuit data in.
        circuit (QuantumCircuit): The circuit data to write.
        metadata_serializer (JSONEncoder): An optional JSONEncoder class that
            will be passed the :attr:`.QuantumCircuit.metadata` dictionary for
            ``circuit`` and will be used as the ``cls`` kwarg
            on the ``json.dump()`` call to JSON serialize that dictionary.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        version (int): The QPY format version to use for serializing this circuit
    """
    metadata_raw = json.dumps(circuit.metadata, separators=(',', ':'), cls=metadata_serializer).encode(common.ENCODE)
    metadata_size = len(metadata_raw)
    num_instructions = len(circuit)
    circuit_name = circuit.name.encode(common.ENCODE)
    global_phase_type, global_phase_data = value.dumps_value(circuit.global_phase)
    with io.BytesIO() as reg_buf:
        num_qregs = _write_registers(reg_buf, circuit.qregs, circuit.qubits)
        num_cregs = _write_registers(reg_buf, circuit.cregs, circuit.clbits)
        registers_raw = reg_buf.getvalue()
    num_registers = num_qregs + num_cregs
    header_raw = formats.CIRCUIT_HEADER_V2(name_size=len(circuit_name), global_phase_type=global_phase_type, global_phase_size=len(global_phase_data), num_qubits=circuit.num_qubits, num_clbits=circuit.num_clbits, metadata_size=metadata_size, num_registers=num_registers, num_instructions=num_instructions)
    header = struct.pack(formats.CIRCUIT_HEADER_V2_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(circuit_name)
    file_obj.write(global_phase_data)
    file_obj.write(metadata_raw)
    file_obj.write(registers_raw)
    instruction_buffer = io.BytesIO()
    custom_operations = {}
    index_map = {}
    index_map['q'] = {bit: index for index, bit in enumerate(circuit.qubits)}
    index_map['c'] = {bit: index for index, bit in enumerate(circuit.clbits)}
    for instruction in circuit.data:
        _write_instruction(instruction_buffer, instruction, custom_operations, index_map, use_symengine, version)
    with io.BytesIO() as custom_operations_buffer:
        new_custom_operations = list(custom_operations.keys())
        while new_custom_operations:
            operations_to_serialize = new_custom_operations.copy()
            new_custom_operations = []
            for name in operations_to_serialize:
                operation = custom_operations[name]
                new_custom_operations.extend(_write_custom_operation(custom_operations_buffer, name, operation, custom_operations, use_symengine, version))
        file_obj.write(struct.pack(formats.CUSTOM_CIRCUIT_DEF_HEADER_PACK, len(custom_operations)))
        file_obj.write(custom_operations_buffer.getvalue())
    file_obj.write(instruction_buffer.getvalue())
    instruction_buffer.close()
    _write_calibrations(file_obj, circuit.calibrations, metadata_serializer)
    _write_layout(file_obj, circuit)