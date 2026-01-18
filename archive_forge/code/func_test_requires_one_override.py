from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_requires_one_override():

    class C(cirq.NoiseModel):
        pass
    with pytest.raises(TypeError, match='abstract'):
        _ = C()