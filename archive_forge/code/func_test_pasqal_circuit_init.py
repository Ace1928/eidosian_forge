from unittest.mock import patch
import copy
import numpy as np
import sympy
import pytest
import cirq
import cirq_pasqal
def test_pasqal_circuit_init():
    qs = cirq.NamedQubit.range(3, prefix='q')
    ex_circuit = cirq.Circuit()
    ex_circuit.append([[cirq.CZ(qs[i], qs[i + 1]), cirq.X(qs[i + 1])] for i in range(len(qs) - 1)])
    test_circuit = cirq.Circuit()
    test_circuit.append([[cirq.CZ(qs[i], qs[i + 1]), cirq.X(qs[i + 1])] for i in range(len(qs) - 1)])
    for moment1, moment2 in zip(test_circuit, ex_circuit):
        assert moment1 == moment2