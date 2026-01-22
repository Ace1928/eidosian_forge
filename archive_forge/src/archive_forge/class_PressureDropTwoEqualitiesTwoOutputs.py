from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropTwoEqualitiesTwoOutputs(ExternalGreyBoxModel):

    def __init__(self):
        self._input_names = ['Pin', 'c', 'F', 'P1', 'P3']
        self._input_values = np.zeros(5, dtype=np.float64)
        self._equality_constraint_names = ['pdrop1', 'pdrop3']
        self._output_names = ['P2', 'Pout']

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values):
        assert len(input_values) == 5
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P1 = self._input_values[3]
        P3 = self._input_values[4]
        return np.asarray([P1 - (Pin - c * F ** 2), P3 - (P1 - 2 * c * F ** 2)], dtype=np.float64)

    def evaluate_outputs(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P1 = self._input_values[3]
        return np.asarray([P1 - c * F ** 2, Pin - 4 * c * F ** 2], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
        nonzeros = np.asarray([-1, F ** 2, 2 * c * F, 1, 2 * F ** 2, 4 * c * F, -1, 1], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2, 5))
        return jac

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([1, 2, 3, 0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray([-F ** 2, -c * 2 * F, 1, 1, -4 * F ** 2, -4 * c * 2 * F], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2, 5))
        return jac