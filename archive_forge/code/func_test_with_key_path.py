import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_with_key_path():
    a = cirq.LineQubit(0)
    op = cirq.measure(a, key='m')
    remap_op = cirq.with_key_path(op, ('a', 'b'))
    assert cirq.measurement_key_names(remap_op) == {'a:b:m'}
    assert cirq.with_key_path(remap_op, ('a', 'b')) is remap_op
    assert cirq.with_key_path(op, tuple()) is op
    assert cirq.with_key_path(cirq.X(a), ('a', 'b')) is NotImplemented