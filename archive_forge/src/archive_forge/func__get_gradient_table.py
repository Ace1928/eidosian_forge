import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
def _get_gradient_table(self):
    bval = np.loadtxt(self.inputs.in_bval)
    bvec = np.loadtxt(self.inputs.in_bvec).T
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bval, bvec)
    gtab.b0_threshold = self.inputs.b0_thres
    return gtab