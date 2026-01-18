import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def x_to_hzh(op: 'cirq.Operation'):
    if isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1:
        return [cirq.H(*op.qubits), cirq.Z(*op.qubits), cirq.H(*op.qubits)]