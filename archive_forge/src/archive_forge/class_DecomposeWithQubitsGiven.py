import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
class DecomposeWithQubitsGiven:

    def __init__(self, func):
        self.func = func

    def _decompose_(self, qubits):
        return self.func(*qubits)