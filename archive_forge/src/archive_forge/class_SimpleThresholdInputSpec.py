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
class SimpleThresholdInputSpec(BaseInterfaceInputSpec):
    volumes = InputMultiPath(File(exists=True), desc='volumes to be thresholded', mandatory=True)
    threshold = traits.Float(desc='volumes to be thresholdedeverything below this value will be set        to zero', mandatory=True)