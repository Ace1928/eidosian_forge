import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_value_to_float():
    assert cg.Calibration.value_to_float([1.1]) == 1.1
    assert cg.Calibration.value_to_float([0.7, 0.5]) == 0.7
    assert cg.Calibration.value_to_float([7]) == 7
    with pytest.raises(ValueError, match='was empty'):
        cg.Calibration.value_to_float([])
    with pytest.raises(ValueError, match='could not convert string to float'):
        cg.Calibration.value_to_float(['went for a walk'])