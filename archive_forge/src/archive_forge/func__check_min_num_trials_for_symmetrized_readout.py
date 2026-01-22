import itertools
import warnings
from math import log, pi
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._compiler import AbstractCompiler, QVMCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3
from pyquil.api._quantum_computer import get_qc as get_qc_v3, QuantumExecutable
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.gates import MEASURE, RX
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.quil import Program, validate_supported_quil
from pyquil.quilatom import qubit_index
from ._qam import StatefulQAM
def _check_min_num_trials_for_symmetrized_readout(num_qubits: int, trials: int, symm_type: int) -> int:
    """
    This function sets the minimum number of trials; it is desirable to have hundreds or
    thousands of trials more than the minimum.

    :param num_qubits: number of qubits to symmetrize
    :param trials: number of trials
    :param symm_type: symmetrization type see
    :return: possibly modified number of trials
    """
    if symm_type < -1 or symm_type > 3:
        raise ValueError('symm_type must be one of the following ints [-1, 0, 1, 2, 3].')
    if symm_type == -1:
        min_num_trials = 2 ** num_qubits
    elif symm_type == 2:

        def _f(x: int) -> int:
            return 4 * x - 1
        min_num_trials = min((_f(x) for x in range(1, 1024) if _f(x) >= num_qubits)) + 1
    elif symm_type == 3:
        min_num_trials = _next_power_of_2(2 * num_qubits)
    else:
        min_num_trials = 2
    if trials < min_num_trials:
        trials = min_num_trials
        warnings.warn(f'Number of trials was too low, it is now {trials}.')
    return trials