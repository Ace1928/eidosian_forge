from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropTwoOutputs(ExternalGreyBoxModel):

    def __init__(self):
        self._input_names = ['Pin', 'c', 'F']
        self._input_values = np.zeros(3, dtype=np.float64)
        self._output_names = ['P2', 'Pout']

    def input_names(self):
        return self._input_names

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values):
        assert len(input_values) == 3
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        raise NotImplementedError('This method should not be called for this model.')

    def evaluate_outputs(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P2 = Pin - 2 * c * F ** 2
        Pout = Pin - 4 * c * F ** 2
        return np.asarray([P2, Pout], dtype=np.float64)

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray([1, -2 * F ** 2, -2 * c * 2 * F, 1, -4 * F ** 2, -4 * c * 2 * F], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2, 3))
        return jac