from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import ClassVar, Optional, Type
import logging
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.weyl import transform_to_magic_basis
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
from qiskit._accelerate import two_qubit_decompose
class _LazyTwoQubitCXDecomposer(TwoQubitBasisDecomposer):
    __slots__ = ('_inner',)

    def __init__(self):
        self._inner = None

    def _load(self):
        if self._inner is None:
            self._inner = TwoQubitBasisDecomposer(CXGate())

    def __call__(self, *args, **kwargs) -> QuantumCircuit:
        self._load()
        return self._inner(*args, **kwargs)

    def traces(self, target):
        self._load()
        return self._inner.traces(target)

    def decomp1(self, target):
        self._load()
        return self._inner.decomp1(target)

    def decomp2_supercontrolled(self, target):
        self._load()
        return self._inner.decomp2_supercontrolled(target)

    def decomp3_supercontrolled(self, target):
        self._load()
        return self._inner.decomp3_supercontrolled(target)

    def num_basis_gates(self, unitary):
        self._load()
        return self._inner.num_basis_gates(unitary)