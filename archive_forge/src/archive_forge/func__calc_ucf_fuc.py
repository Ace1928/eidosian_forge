import warnings
import numpy as np
from .layer import (
from .pmatrix import PMatrix
from ..cnot_unit_objective import CNOTUnitObjective
def _calc_ucf_fuc(self):
    """
        Computes matrices ``ucf_mat`` and ``fuc_mat``. Both remain non-finalized.
        """
    ucf_mat = self._ucf_mat
    fuc_mat = self._fuc_mat
    tmp1 = self._tmp1
    c_layers = self._c_layers
    f_layers = self._f_layers
    depth, n = (self.num_cnots, self._num_qubits)
    np.conj(self.target_matrix.T, out=tmp1)
    self._ucf_mat.set_matrix(tmp1)
    for q in range(depth - 1, -1, -1):
        ucf_mat.mul_right_q2(c_layers[q], temp_mat=tmp1, dagger=False)
    fuc_mat.set_matrix(ucf_mat.finalize(temp_mat=tmp1))
    for q in range(n):
        fuc_mat.mul_left_q1(f_layers[q], temp_mat=tmp1)
    for q in range(n - 1, -1, -1):
        ucf_mat.mul_right_q1(f_layers[q], temp_mat=tmp1, dagger=False)