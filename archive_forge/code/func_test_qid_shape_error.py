from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_qid_shape_error():
    with pytest.raises(ValueError, match='qid_shape must be provided'):
        cirq.sim.state_vector_simulation_state._BufferedStateVector.create(initial_state=0)