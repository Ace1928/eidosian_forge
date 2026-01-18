from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, SupportsFloat
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
@property
@abstractmethod
def thetas(self) -> np.ndarray:
    """
        The property is not implemented and raises a ``NotImplementedException`` exception.

        Returns:
            a vector of parameters of this circuit.
        """
    raise NotImplementedError