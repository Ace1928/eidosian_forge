from typing import (
import numpy as np
from cirq import protocols, qis, value
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _extend_matrix(self, sub_matrix: np.ndarray) -> np.ndarray:
    qid_shape = protocols.qid_shape(self)
    sub_n = len(qid_shape) - len(self.controls)
    tensor = qis.eye_tensor(qid_shape, dtype=sub_matrix.dtype)
    sub_tensor = sub_matrix.reshape(qid_shape[len(self.controls):] * 2)
    for control_vals in self.control_values.expand():
        active = (*(v for v in control_vals), *(slice(None),) * sub_n) * 2
        tensor[active] = sub_tensor
    return tensor.reshape((np.prod(qid_shape, dtype=np.int64).item(),) * 2)