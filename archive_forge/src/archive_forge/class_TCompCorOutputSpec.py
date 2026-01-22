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
class TCompCorOutputSpec(CompCorOutputSpec):
    high_variance_masks = OutputMultiPath(File(exists=True), desc='voxels exceeding the variance threshold')