from typing import Tuple, Optional, List, Union, Generic, TypeVar, Dict
from unittest.mock import create_autospec, Mock
import pytest
from pyquil import Program
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.api import QAM, QuantumComputer, QuantumExecutable, QAMExecutionResult, EncryptedProgram
from pyquil.api._abstract_compiler import AbstractCompiler
from qcs_api_client.client._configuration.settings import QCSClientConfigurationSettings
from qcs_api_client.client._configuration import (
import networkx as nx
import cirq
import sympy
import numpy as np
class MockQPUImplementer:

    def __init__(self, quantum_computer: QuantumComputer):
        """Initializes a MockQPUImplementer.

        Args:
            quantum_computer: QuantumComputer to mock.
        """
        self.quantum_computer = quantum_computer

    def implement_passive_quantum_computer_with_results(self, results: List[np.ndarray]) -> QuantumComputer:
        """Mocks compilation methods on the `quantum_computer.compiler`, passively passing the
        `Program` through. Sequentially adds results to the
        `quantum_computer.qam._memory_region` (this will not work for asynchronous runs).

        Args:
            results: np.ndarray to sequentially write to `QAM._memory_region`.

        Returns:
            A mocked QuantumComputer.
        """
        quantum_computer = self.quantum_computer

        def quil_to_native_quil(program: Program, *, protoquil: Optional[bool]=None) -> Program:
            return program
        quantum_computer.compiler.quil_to_native_quil = create_autospec(quantum_computer.compiler.quil_to_native_quil, side_effect=quil_to_native_quil)

        def native_quil_to_executable(nq_program: Program) -> QuantumExecutable:
            assert 2 == nq_program.num_shots
            return nq_program
        quantum_computer.compiler.native_quil_to_executable = create_autospec(quantum_computer.compiler.native_quil_to_executable, side_effect=native_quil_to_executable)

        def run(program: Union[Program, EncryptedProgram]) -> QAMExecutionResult:
            qam = quantum_computer.qam
            qam._mock_results = qam._mock_results or {}
            qam._mock_results['m0'] = results[qam._run_count]
            quantum_computer.qam._run_count += 1
            return QAMExecutionResult(executable=program, readout_data=qam._mock_results)
        quantum_computer.qam.run = Mock(quantum_computer.qam.run, side_effect=run)
        return quantum_computer