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
def validate_supported_quil(program: Program) -> None:
    """
    Ensure that a program is supported Quil which can run on any QPU, otherwise raise a ValueError.
    We support a global RESET before any gates, and MEASUREs on each qubit after any gates
    on that qubit. PRAGMAs and DECLAREs are always allowed.

    :param program: The Quil program to validate.
    """
    gates_seen = False
    measured_qubits: Set[int] = set()
    for instr in program.instructions:
        if isinstance(instr, Pragma) or isinstance(instr, Declare):
            continue
        elif isinstance(instr, Gate):
            gates_seen = True
            if any((q.index in measured_qubits for q in instr.qubits)):
                raise ValueError('Cannot apply gates to qubits that were already measured.')
        elif isinstance(instr, Reset):
            if gates_seen:
                raise ValueError('RESET can only be applied before any gate applications.')
        elif isinstance(instr, ResetQubit):
            raise ValueError('Only global RESETs are currently supported.')
        elif isinstance(instr, Measurement):
            if instr.qubit.index in measured_qubits:
                raise ValueError('Multiple measurements per qubit is not supported.')
            measured_qubits.add(instr.qubit.index)
        else:
            raise ValueError(f'Unhandled instruction type in supported Quil validation: {instr}')