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
def _write_layout(file_obj, circuit):
    if circuit.layout is None:
        file_obj.write(struct.pack(formats.LAYOUT_V2_PACK, False, -1, -1, -1, 0, 0))
        return
    initial_size = -1
    input_qubit_mapping = {}
    initial_layout_array = []
    extra_registers = defaultdict(list)
    if circuit.layout.initial_layout is not None:
        initial_size = len(circuit.layout.initial_layout)
        layout_mapping = circuit.layout.initial_layout.get_physical_bits()
        for i in range(circuit.num_qubits):
            qubit = layout_mapping[i]
            input_qubit_mapping[qubit] = i
            if qubit._register is not None or qubit._index is not None:
                if qubit._register not in circuit.qregs:
                    extra_registers[qubit._register].append(qubit)
                initial_layout_array.append((qubit._index, qubit._register))
            else:
                initial_layout_array.append((None, None))
    input_qubit_size = -1
    input_qubit_mapping_array = []
    if circuit.layout.input_qubit_mapping is not None:
        input_qubit_size = len(circuit.layout.input_qubit_mapping)
        input_qubit_mapping_array = [None] * input_qubit_size
        layout_mapping = circuit.layout.initial_layout.get_virtual_bits()
        for qubit, index in circuit.layout.input_qubit_mapping.items():
            if getattr(qubit, '_register', None) is not None and getattr(qubit, '_index', None) is not None:
                if qubit._register not in circuit.qregs:
                    extra_registers[qubit._register].append(qubit)
                input_qubit_mapping_array[index] = layout_mapping[qubit]
            else:
                input_qubit_mapping_array[index] = layout_mapping[qubit]
    final_layout_size = -1
    final_layout_array = []
    if circuit.layout.final_layout is not None:
        final_layout_size = len(circuit.layout.final_layout)
        final_layout_physical = circuit.layout.final_layout.get_physical_bits()
        for i in range(circuit.num_qubits):
            virtual_bit = final_layout_physical[i]
            final_layout_array.append(circuit.find_bit(virtual_bit).index)
    input_qubit_count = circuit._layout._input_qubit_count
    if input_qubit_count is None:
        input_qubit_count = -1
    file_obj.write(struct.pack(formats.LAYOUT_V2_PACK, True, initial_size, input_qubit_size, final_layout_size, len(extra_registers), input_qubit_count))
    _write_registers(file_obj, list(extra_registers), [x for bits in extra_registers.values() for x in bits])
    for index, register in initial_layout_array:
        reg_name_bytes = None if register is None else register.name.encode(common.ENCODE)
        file_obj.write(struct.pack(formats.INITIAL_LAYOUT_BIT_PACK, -1 if index is None else index, -1 if reg_name_bytes is None else len(reg_name_bytes)))
        if reg_name_bytes is not None:
            file_obj.write(reg_name_bytes)
    for i in input_qubit_mapping_array:
        file_obj.write(struct.pack('!I', i))
    for i in final_layout_array:
        file_obj.write(struct.pack('!I', i))