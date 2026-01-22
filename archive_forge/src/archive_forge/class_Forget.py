from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
class Forget(cirq.NoiseModel):

    def noisy_operation(self, operation):
        raise NotImplementedError()