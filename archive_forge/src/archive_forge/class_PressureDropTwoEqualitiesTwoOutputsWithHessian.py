from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropTwoEqualitiesTwoOutputsWithHessian(PressureDropTwoEqualitiesTwoOutputs):

    def __init__(self):
        super(PressureDropTwoEqualitiesTwoOutputsWithHessian, self).__init__()
        self._eq_con_mult_values = np.zeros(2, dtype=np.float64)
        self._output_con_mult_values = np.zeros(2, dtype=np.float64)

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 2
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 2
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_hessian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        y1 = self._eq_con_mult_values[0]
        y2 = self._eq_con_mult_values[1]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        nonzeros = np.asarray([y1 * (2 * F) + y2 * (4 * F), y1 * (2 * c) + y2 * (4 * c)], dtype=np.float64)
        hess = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(5, 5))
        return hess

    def evaluate_hessian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        y1 = self._output_con_mult_values[0]
        y2 = self._output_con_mult_values[1]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        nonzeros = np.asarray([y1 * (-2 * F) + y2 * (-8 * F), y1 * (-2 * c) + y2 * (-8 * c)], dtype=np.float64)
        hess = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(5, 5))
        return hess