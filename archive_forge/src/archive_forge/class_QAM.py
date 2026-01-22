from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Mapping, Optional, TypeVar
import numpy as np
from pyquil.api._abstract_compiler import QuantumExecutable
class QAM(ABC, Generic[T]):
    """
    Quantum Abstract Machine: This class acts as a generic interface describing how a classical
    computer interacts with a live quantum computer.
    """

    @abstractmethod
    def execute(self, executable: QuantumExecutable) -> T:
        """
        Run an executable on a QAM, returning a handle to be used to retrieve
        results.

        :param executable: The executable program to be executed by the QAM.
        """

    @abstractmethod
    def get_result(self, execute_response: T) -> QAMExecutionResult:
        """
        Retrieve the results associated with a previous call to ``QAM#execute``.

        :param execute_response: The return value from a call to ``execute``.
        """

    def run(self, executable: QuantumExecutable) -> QAMExecutionResult:
        """
        Run an executable to completion on the QAM.
        """
        return self.get_result(self.execute(executable))