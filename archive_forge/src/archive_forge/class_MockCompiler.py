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
class MockCompiler(AbstractCompiler):

    def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool]=None) -> Program:
        raise NotImplementedError

    def native_quil_to_executable(self, nq_program: Program) -> QuantumExecutable:
        raise NotImplementedError