import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@functools.lru_cache()
def _all_clifford_gates() -> Tuple['cirq.SingleQubitCliffordGate', ...]:
    return tuple((cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z) for trans_x, trans_z in _all_rotation_pairs()))