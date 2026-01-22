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
def _read_registers(file_obj, num_registers):
    registers = {'q': {}, 'c': {}}
    for _reg in range(num_registers):
        data = formats.REGISTER._make(struct.unpack(formats.REGISTER_PACK, file_obj.read(formats.REGISTER_SIZE)))
        name = file_obj.read(data.name_size).decode('utf8')
        REGISTER_ARRAY_PACK = '!%sI' % data.size
        bit_indices_raw = file_obj.read(struct.calcsize(REGISTER_ARRAY_PACK))
        bit_indices = list(struct.unpack(REGISTER_ARRAY_PACK, bit_indices_raw))
        if data.type.decode('utf8') == 'q':
            registers['q'][name] = (data.standalone, bit_indices, True)
        else:
            registers['c'][name] = (data.standalone, bit_indices, True)
    return registers