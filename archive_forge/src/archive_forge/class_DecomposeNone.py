import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
class DecomposeNone:

    def _decompose_(self, qubits=None):
        return None