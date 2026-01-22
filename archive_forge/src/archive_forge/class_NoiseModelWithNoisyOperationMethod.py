from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
class NoiseModelWithNoisyOperationMethod(cirq.NoiseModel):

    def noisy_operation(self, operation: 'cirq.Operation'):
        return cirq.Z(operation.qubits[0]).with_tags(ops.VirtualTag())