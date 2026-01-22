import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
def _bool_vec_dissimilarity(self, booldata1, booldata2, method):
    from scipy.spatial.distance import dice, jaccard
    methods = {'dice': dice, 'jaccard': jaccard}
    if not (np.any(booldata1) or np.any(booldata2)):
        return 0
    return 1 - methods[method](booldata1.flat, booldata2.flat)