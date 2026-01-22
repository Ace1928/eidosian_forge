import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def apply_noise_model(prog: 'Program', noise_model: NoiseModel) -> 'Program':
    """
    Apply a noise model to a program and generated a 'noisy-fied' version of the program.

    :param prog: A Quil Program object.
    :param noise_model: A NoiseModel, either generated from an ISA or
        from a simple decoherence model.
    :return: A new program translated to a noisy gateset and with noisy readout as described by the
        noisemodel.
    """
    new_prog = _noise_model_program_header(noise_model)
    for i in prog:
        if isinstance(i, Gate) and noise_model.gates:
            try:
                _, new_name = get_noisy_gate(i.name, tuple(i.params))
                new_prog += Gate(new_name, [], i.qubits)
            except NoisyGateUndefined:
                new_prog += i
        else:
            new_prog += i
    return prog.copy_everything_except_instructions() + new_prog