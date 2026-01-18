import pytest
import numpy as np
from traitlets import TraitError, Undefined
from ..ndarray.traits import shape_constraints
from ..ndarray.widgets import (
def test_notification(mock_comm):
    data = np.zeros((2, 4))
    w = NDArrayWidget(data)
    w.comm = mock_comm
    w.array = np.ones((2, 4, 2))
    assert len(mock_comm.log_send) == 1