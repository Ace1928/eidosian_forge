from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_tagged_operation_equality():
    eq = cirq.testing.EqualsTester()
    q1 = cirq.GridQubit(1, 1)
    op = cirq.X(q1)
    op2 = cirq.Y(q1)
    eq.add_equality_group(op)
    eq.add_equality_group(op.with_tags('tag1'), cirq.TaggedOperation(op, 'tag1'))
    eq.add_equality_group(op2.with_tags('tag1'), cirq.TaggedOperation(op2, 'tag1'))
    eq.add_equality_group(op.with_tags('tag2'), cirq.TaggedOperation(op, 'tag2'))
    eq.add_equality_group(op.with_tags('tag1', 'tag2'), op.with_tags('tag1').with_tags('tag2'), cirq.TaggedOperation(op, 'tag1', 'tag2'))