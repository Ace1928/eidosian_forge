import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
@no_type_check
def address_qubits(program: Program, qubit_mapping: Optional[Dict[QubitPlaceholder, Union[Qubit, int]]]=None) -> Program:
    """
    Takes a program which contains placeholders and assigns them all defined values.

    Either all qubits must be defined or all undefined. If qubits are
    undefined, you may provide a qubit mapping to specify how placeholders get mapped
    to actual qubits. If a mapping is not provided, integers 0 through N are used.

    This function will also instantiate any label placeholders.

    :param program: The program.
    :param qubit_mapping: A dictionary-like object that maps from :py:class:`QubitPlaceholder`
        to :py:class:`Qubit` or ``int`` (but not both).
    :return: A new Program with all qubit and label placeholders assigned to real qubits and labels.
    """
    fake_qubits, real_qubits, qubits = _what_type_of_qubit_does_it_use(program)
    if real_qubits:
        if qubit_mapping is not None:
            warnings.warn('A qubit mapping was provided but the program does not contain any placeholders to map!')
        return program
    if qubit_mapping is None:
        qubit_mapping = {qp: Qubit(i) for i, qp in enumerate(qubits)}
    elif all((isinstance(v, Qubit) for v in qubit_mapping.values())):
        pass
    elif all((isinstance(v, int) for v in qubit_mapping.values())):
        qubit_mapping = {k: Qubit(v) for k, v in qubit_mapping.items()}
    else:
        raise ValueError('Qubit mapping must map to type Qubit or int (but not both)')
    result: List[AbstractInstruction] = []
    for instr in program:
        if isinstance(instr, Gate):
            remapped_qubits = [qubit_mapping[q] for q in instr.qubits]
            gate = Gate(instr.name, instr.params, remapped_qubits)
            gate.modifiers = instr.modifiers
            result.append(gate)
        elif isinstance(instr, Measurement):
            result.append(Measurement(qubit_mapping[instr.qubit], instr.classical_reg))
        elif isinstance(instr, ResetQubit):
            result.append(ResetQubit(qubit_mapping[instr.qubit]))
        elif isinstance(instr, Pragma):
            new_args: List[Union[Qubit, int, str]] = []
            for arg in instr.args:
                if isinstance(arg, QubitPlaceholder):
                    new_args.append(qubit_mapping[arg])
                else:
                    new_args.append(arg)
            result.append(Pragma(instr.command, new_args, instr.freeform_string))
        else:
            result.append(instr)
    new_program = program.copy()
    new_program._instructions = result
    return new_program