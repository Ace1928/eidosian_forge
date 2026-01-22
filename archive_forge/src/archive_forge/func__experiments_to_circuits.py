from typing import Any, Dict, List, NewType, Tuple, Union
import collections
import math
from qiskit import pulse
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
def _experiments_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj.

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits

    Returns:
        list: A list of QuantumCircuit objects from the qobj
    """
    if not qobj.experiments:
        return None
    circuits = []
    for exp in qobj.experiments:
        quantum_registers = [QuantumRegister(i[1], name=i[0]) for i in exp.header.qreg_sizes]
        classical_registers = [ClassicalRegister(i[1], name=i[0]) for i in exp.header.creg_sizes]
        circuit = QuantumCircuit(*quantum_registers, *classical_registers, name=exp.header.name)
        qreg_dict = collections.OrderedDict()
        creg_dict = collections.OrderedDict()
        for reg in quantum_registers:
            qreg_dict[reg.name] = reg
        for reg in classical_registers:
            creg_dict[reg.name] = reg
        conditional = {}
        for i in exp.instructions:
            name = i.name
            qubits = []
            params = getattr(i, 'params', [])
            try:
                for qubit in i.qubits:
                    qubit_label = exp.header.qubit_labels[qubit]
                    qubits.append(qreg_dict[qubit_label[0]][qubit_label[1]])
            except Exception:
                pass
            clbits = []
            try:
                for clbit in i.memory:
                    clbit_label = exp.header.clbit_labels[clbit]
                    clbits.append(creg_dict[clbit_label[0]][clbit_label[1]])
            except Exception:
                pass
            if hasattr(circuit, name):
                instr_method = getattr(circuit, name)
                if i.name == 'initialize':
                    _inst = instr_method(params, qubits)
                elif i.name in ['mcx', 'mcu1', 'mcp']:
                    _inst = instr_method(*params, qubits[:-1], qubits[-1], *clbits)
                else:
                    _inst = instr_method(*params, *qubits, *clbits)
            elif name == 'bfunc':
                conditional['value'] = int(i.val, 16)
                full_bit_size = sum((creg_dict[x].size for x in creg_dict))
                mask_map = {}
                raw_map = {}
                raw = []
                for creg in creg_dict:
                    size = creg_dict[creg].size
                    reg_raw = [1] * size
                    if not raw:
                        raw = reg_raw
                    else:
                        for pos, val in enumerate(raw):
                            if val == 1:
                                raw[pos] = 0
                        raw = reg_raw + raw
                    mask = [0] * (full_bit_size - len(raw)) + raw
                    raw_map[creg] = mask
                    mask_map[int(''.join((str(x) for x in mask)), 2)] = creg
                if bin(int(i.mask, 16)).count('1') == 1:
                    cbit = int(math.log2(int(i.mask, 16)))
                    for reg in creg_dict.values():
                        size = reg.size
                        if cbit >= size:
                            cbit -= size
                        else:
                            conditional['register'] = reg if reg.size == 1 else reg[cbit]
                            break
                    mask_str = bin(int(i.mask, 16))[2:].zfill(full_bit_size)
                    mask = [int(item) for item in list(mask_str)]
                else:
                    creg = mask_map[int(i.mask, 16)]
                    conditional['register'] = creg_dict[creg]
                    mask = raw_map[creg]
                val = int(i.val, 16)
                for j in reversed(mask):
                    if j == 0:
                        val = val >> 1
                    else:
                        conditional['value'] = val
                        break
            else:
                _inst = temp_opaque_instruction = Instruction(name=name, num_qubits=len(qubits), num_clbits=len(clbits), params=params)
                circuit.append(temp_opaque_instruction, qubits, clbits)
            if conditional and name != 'bfunc':
                _inst.c_if(conditional['register'], conditional['value'])
                conditional = {}
        pulse_lib = qobj.config.pulse_library if hasattr(qobj.config, 'pulse_library') else []
        if hasattr(qobj.config, 'calibrations'):
            circuit.calibrations = dict(**circuit.calibrations, **_qobj_to_circuit_cals(qobj, pulse_lib))
        if hasattr(exp.config, 'calibrations'):
            circuit.calibrations = dict(**circuit.calibrations, **_qobj_to_circuit_cals(exp, pulse_lib))
        circuits.append(circuit)
    return circuits