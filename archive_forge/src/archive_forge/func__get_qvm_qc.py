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
def _get_qvm_qc(*, client_configuration: QCSClientConfiguration, name: str, qvm_type: str, quantum_processor: AbstractQuantumProcessor, compiler_timeout: float, execution_timeout: float, noise_model: Optional[NoiseModel]) -> QuantumComputer:
    """Construct a QuantumComputer backed by a QVM.

    This is a minimal wrapper over the QuantumComputer, QVM, and QVMCompiler constructors.

    :param client_configuration: Client configuration.
    :param name: A string identifying this particular quantum computer.
    :param qvm_type: The type of QVM. Either qvm or pyqvm.
    :param quantum_processor: A quantum_processor following the AbstractQuantumProcessor interface.
    :param noise_model: An optional noise model
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :return: A QuantumComputer backed by a QVM with the above options.
    """
    return QuantumComputer(name=name, qam=_get_qvm_or_pyqvm(client_configuration=client_configuration, qvm_type=qvm_type, noise_model=noise_model, quantum_processor=quantum_processor, execution_timeout=execution_timeout), compiler=QVMCompiler(quantum_processor=quantum_processor, timeout=compiler_timeout, client_configuration=client_configuration))