from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropSingleEqualityWithHessian(PressureDropSingleEquality):

    def __init__(self):
        super(PressureDropSingleEqualityWithHessian, self).__init__()
        self._eq_con_mult_values = np.zeros(1, dtype=np.float64)

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 1
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)

    def evaluate_hessian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        nonzeros = self._eq_con_mult_values[0] * np.asarray([8 * F, 8 * c], dtype=np.float64)
        hess = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(4, 4))
        return hess