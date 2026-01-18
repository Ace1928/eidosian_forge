import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def test_random_device_placer_repr():
    cirq.testing.assert_equivalent_repr(cg.RandomDevicePlacer(), global_vals={'cirq_google': cg})