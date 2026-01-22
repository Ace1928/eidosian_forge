import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
def _compute_tSTD(M, x, axis=0):
    stdM = np.std(M, axis=axis)
    stdM[stdM == 0] = x
    stdM[np.isnan(stdM)] = x
    return stdM