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
class CalculateNormalizedMomentsInputSpec(TraitedSpec):
    timeseries_file = File(exists=True, mandatory=True, desc='Text file with timeseries in columns and timepoints in rows,        whitespace separated')
    moment = traits.Int(mandatory=True, desc='Define which moment should be calculated, 3 for skewness, 4 for        kurtosis.')