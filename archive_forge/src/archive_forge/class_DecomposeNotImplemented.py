import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
class DecomposeNotImplemented:

    def _decompose_(self, qubits=None):
        return NotImplemented