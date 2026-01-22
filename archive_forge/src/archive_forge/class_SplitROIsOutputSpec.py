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
class SplitROIsOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(File(exists=True), desc='the resulting ROIs')
    out_masks = OutputMultiPath(File(exists=True), desc='a mask indicating valid values')
    out_index = OutputMultiPath(File(exists=True), desc='arrays keeping original locations')