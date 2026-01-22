from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropTwoEqualitiesTwoOutputsScaleEqualities(PressureDropTwoEqualitiesTwoOutputs):

    def get_equality_constraint_scaling_factors(self):
        return np.asarray([3.1, 3.2], dtype=np.float64)