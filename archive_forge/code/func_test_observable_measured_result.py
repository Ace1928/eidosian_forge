import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_observable_measured_result():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    omr = cw.ObservableMeasuredResult(setting=cw.InitObsSetting(init_state=cirq.Z(a) * cirq.Z(b), observable=cirq.Y(a) * cirq.Y(b)), mean=0, variance=5 ** 2, repetitions=4, circuit_params={'phi': 52})
    assert omr.stddev == 5
    assert omr.observable == cirq.Y(a) * cirq.Y(b)
    assert omr.init_state == cirq.Z(a) * cirq.Z(b)
    cirq.testing.assert_equivalent_repr(omr)
    assert omr.as_dict() == {'init_state': cirq.Z(a) * cirq.Z(b), 'observable': cirq.Y(a) * cirq.Y(b), 'mean': 0, 'variance': 25, 'repetitions': 4, 'param.phi': 52}
    omr2 = dataclasses.replace(omr, circuit_params={'phi': 52, 'observable': 3.14, 'param.phi': -1})
    assert omr2.as_dict() == {'init_state': cirq.Z(a) * cirq.Z(b), 'observable': cirq.Y(a) * cirq.Y(b), 'mean': 0, 'variance': 25, 'repetitions': 4, 'param.phi': 52, 'param.observable': 3.14, 'param.param.phi': -1}