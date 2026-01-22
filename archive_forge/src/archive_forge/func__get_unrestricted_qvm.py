import itertools
import re
import socket
import subprocess
import warnings
from contextlib import contextmanager
from math import pi, log
from typing import (
import httpx
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models import ListQuantumProcessorsResponse
from qcs_api_client.operations.sync import list_quantum_processors
from rpcq.messages import ParameterAref
from pyquil.api import EngagementManager
from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable
from pyquil.api._compiler import QPUCompiler, QVMCompiler
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qcs_client import qcs_client
from pyquil.api._qpu import QPU
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import RX, MEASURE
from pyquil.noise import decoherence_noise_with_asymmetric_ro, NoiseModel
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import (
from pyquil.quil import Program
def _get_unrestricted_qvm(*, client_configuration: QCSClientConfiguration, name: str, noisy: bool, n_qubits: int, qvm_type: str, compiler_timeout: float, execution_timeout: float) -> QuantumComputer:
    """
    A qvm with a fully-connected topology.

    This is obviously the least realistic QVM, but who am I to tell users what they want.

    :param client_configuration: Client configuration.
    :param name: The name of this QVM
    :param noisy: Whether to construct a noisy quantum computer
    :param n_qubits: 34 qubits ought to be enough for anybody.
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :return: A pre-configured QuantumComputer
    """
    topology = nx.complete_graph(n_qubits)
    return _get_qvm_with_topology(client_configuration=client_configuration, name=name, topology=topology, noisy=noisy, qvm_type=qvm_type, compiler_timeout=compiler_timeout, execution_timeout=execution_timeout)