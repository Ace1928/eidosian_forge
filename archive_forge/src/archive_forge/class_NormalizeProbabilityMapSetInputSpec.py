import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class NormalizeProbabilityMapSetInputSpec(TraitedSpec):
    in_files = InputMultiPath(File(exists=True, mandatory=True, desc='The tpms to be normalized'))
    in_mask = File(exists=True, desc='Masked voxels must sum up 1.0, 0.0 otherwise.')