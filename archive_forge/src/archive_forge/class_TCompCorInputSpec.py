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
class TCompCorInputSpec(CompCorInputSpec):
    percentile_threshold = traits.Range(low=0.0, high=1.0, value=0.02, exclude_low=True, exclude_high=True, usedefault=True, desc='the percentile used to select highest-variance voxels, represented by a number between 0 and 1, exclusive. By default, this value is set to .02. That is, the 2% of voxels with the highest variance are used.')