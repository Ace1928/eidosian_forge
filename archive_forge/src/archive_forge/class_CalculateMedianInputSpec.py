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
class CalculateMedianInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True, mandatory=True, desc='One or more realigned Nifti 4D timeseries'))
    median_file = traits.Str(desc='Filename prefix to store median images')
    median_per_file = traits.Bool(False, usedefault=True, desc='Calculate a median file for each Nifti')