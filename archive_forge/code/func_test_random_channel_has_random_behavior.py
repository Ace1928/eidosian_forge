from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_random_channel_has_random_behavior():
    q = cirq.LineQubit(0)
    s = cirq.Simulator().sample(cirq.Circuit(cirq.X(q), cirq.amplitude_damp(0.4).on(q), cirq.measure(q, key='out')), repetitions=100)
    v = s['out'].value_counts()
    assert v[0] > 1
    assert v[1] > 1