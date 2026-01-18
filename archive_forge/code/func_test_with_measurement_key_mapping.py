import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_with_measurement_key_mapping():
    a = cirq.LineQubit(0)
    op = cirq.measure(a, key='m')
    remap_op = cirq.with_measurement_key_mapping(op, {'m': 'k'})
    assert cirq.measurement_key_names(remap_op) == {'k'}
    assert cirq.with_measurement_key_mapping(op, {'x': 'k'}) is op