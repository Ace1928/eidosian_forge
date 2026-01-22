import warnings
import numpy as np
from .layer import (
from .pmatrix import PMatrix
from ..cnot_unit_objective import CNOTUnitObjective
def _calc_gradient4d(self, grad4d: np.ndarray):
    """
        Calculates a part gradient contributed by 2-qubit gates.
        """
    fuc = self._fuc_mat
    tmp1, tmp2 = (self._tmp1, self._tmp2)
    c_gates = self._c_gates
    c_dervs = self._c_dervs
    c_layers = self._c_layers
    for q in range(self.num_cnots):
        if q > 0:
            c_layers[q - 1].set_from_matrix(mat=c_gates[q - 1])
            fuc.mul_left_q2(c_layers[q - 1], temp_mat=tmp1)
        fuc.mul_right_q2(c_layers[q], temp_mat=tmp1, dagger=True)
        fuc.finalize(temp_mat=tmp1)
        for i in range(4):
            c_layers[q].set_from_matrix(mat=c_dervs[q, i])
            grad4d[q, i] = -1 * np.real(fuc.product_q2(layer=c_layers[q], tmp1=tmp1, tmp2=tmp2))