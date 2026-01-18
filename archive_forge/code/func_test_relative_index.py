import itertools
import numpy as np
import pytest
import cirq
def test_relative_index():
    assert cirq.X.relative_index(cirq.X) == 0
    assert cirq.X.relative_index(cirq.Y) == -1
    assert cirq.X.relative_index(cirq.Z) == 1
    assert cirq.Y.relative_index(cirq.X) == 1
    assert cirq.Y.relative_index(cirq.Y) == 0
    assert cirq.Y.relative_index(cirq.Z) == -1
    assert cirq.Z.relative_index(cirq.X) == -1
    assert cirq.Z.relative_index(cirq.Y) == 1
    assert cirq.Z.relative_index(cirq.Z) == 0