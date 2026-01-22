from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
@register_model(IntervalType)
class IntervalModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('lo', types.float64), ('hi', types.float64)]
        models.StructModel.__init__(self, dmm, fe_type, members)