from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_tagged_measurement():
    assert not cirq.is_measurement(cirq.global_phase_operation(coefficient=-1.0).with_tags('tag0'))
    a = cirq.LineQubit(0)
    op = cirq.measure(a, key='m').with_tags('tag')
    assert cirq.is_measurement(op)
    remap_op = cirq.with_measurement_key_mapping(op, {'m': 'k'})
    assert remap_op.tags == ('tag',)
    assert cirq.is_measurement(remap_op)
    assert cirq.measurement_key_names(remap_op) == {'k'}
    assert cirq.with_measurement_key_mapping(op, {'x': 'k'}) == op