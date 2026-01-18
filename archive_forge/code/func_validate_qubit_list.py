from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union, Tuple
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil._version import pyquil_version
from pyquil.api import QuantumExecutable
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qvm_client import (
from pyquil.noise import NoiseModel, apply_noise_model
from pyquil.quil import Program, get_classical_addresses_from_program
def validate_qubit_list(qubit_list: Sequence[int]) -> Sequence[int]:
    """
    Check the validity of qubits for the payload.

    :param qubit_list: List of qubits to be validated.
    """
    if not isinstance(qubit_list, Sequence):
        raise TypeError("'qubit_list' must be of type 'Sequence'")
    if any((not isinstance(i, int) or i < 0 for i in qubit_list)):
        raise TypeError("'qubit_list' must contain positive integer values")
    return qubit_list