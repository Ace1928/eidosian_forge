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
def _measure_bitstrings(qc: QuantumComputer, programs: List[Program], meas_qubits: List[int], num_shots: int=600) -> List[np.ndarray]:
    """
    Wrapper for appending measure instructions onto each program, running the program,
    and accumulating the resulting bitarrays.

    :param qc: a quantum computer object on which to run each program
    :param programs: a list of programs to run
    :param meas_qubits: groups of qubits to measure for each program
    :param num_shots: the number of shots to run for each program
    :return: a len(programs) long list of num_shots by num_meas_qubits bit arrays of results for
        each program.
    """
    results = []
    for program in programs:
        prog = program.copy()
        ro = prog.declare('ro', 'BIT', len(meas_qubits))
        for idx, q in enumerate(meas_qubits):
            prog += MEASURE(q, ro[idx])
        prog.wrap_in_numshots_loop(num_shots)
        prog = qc.compiler.quil_to_native_quil(prog)
        executable = qc.compiler.native_quil_to_executable(prog)
        shots = qc.run(executable)
        results.append(shots)
    return results