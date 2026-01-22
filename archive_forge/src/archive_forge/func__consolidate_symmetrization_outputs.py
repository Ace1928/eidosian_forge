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
def _consolidate_symmetrization_outputs(outputs: List[np.ndarray], flip_arrays: List[Tuple[bool]]) -> np.ndarray:
    """
    Given bitarray results from a series of symmetrization programs, appropriately flip output
    bits and consolidate results into new bitarrays.

    :param outputs: a list of the raw bitarrays resulting from running a list of symmetrized
        programs; for example, the results returned from _measure_bitstrings
    :param flip_arrays: a list of boolean arrays in one-to-one correspondence with the list of
        outputs indicating which qubits where flipped before each bitarray was measured.
    :return: an np.ndarray consisting of the consolidated bitarray outputs which can be treated as
        the symmetrized outputs of the original programs passed into a symmetrization method. See
        estimate_observables for example usage.
    """
    assert len(outputs) == len(flip_arrays)
    output = []
    for bitarray, flip_array in zip(outputs, flip_arrays):
        if len(flip_array) == 0:
            output.append(bitarray)
        else:
            output.append(bitarray ^ flip_array)
    return np.vstack(output)