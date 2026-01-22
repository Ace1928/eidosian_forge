import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def _noise_model_program_header(noise_model: NoiseModel) -> 'Program':
    """
    Generate the header for a pyquil Program that uses ``noise_model`` to overload noisy gates.
    The program header consists of 3 sections:

        - The ``DEFGATE`` statements that define the meaning of the newly introduced "noisy" gate
          names.
        - The ``PRAGMA ADD-KRAUS`` statements to overload these noisy gates on specific qubit
          targets with their noisy implementation.
        - THe ``PRAGMA READOUT-POVM`` statements that define the noisy readout per qubit.

    :param noise_model: The assumed noise model.
    :return: A quil Program with the noise pragmas.
    """
    from pyquil.quil import Program
    p = Program()
    defgates: Set[str] = set()
    for k in noise_model.gates:
        try:
            ideal_gate, new_name = get_noisy_gate(k.gate, tuple(k.params))
            if new_name not in defgates:
                p.defgate(new_name, ideal_gate)
                defgates.add(new_name)
        except NoisyGateUndefined:
            print('WARNING: Could not find ideal gate definition for gate {}'.format(k.gate), file=sys.stderr)
            new_name = k.gate
        p.define_noisy_gate(new_name, k.targets, k.kraus_ops)
    for q, ap in noise_model.assignment_probs.items():
        p.define_noisy_readout(q, p00=ap[0, 0], p11=ap[1, 1])
    return p