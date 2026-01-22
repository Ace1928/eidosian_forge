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
def _write_nifti(self, img, data, idx, suffix='median'):
    if self.inputs.median_per_file:
        median_img = nb.Nifti1Image(data, img.affine, img.header)
        filename = self._gen_fname(suffix, idx=idx)
    else:
        median_img = nb.Nifti1Image(data / (idx + 1), img.affine, img.header)
        filename = self._gen_fname(suffix)
    median_img.to_filename(filename)
    return filename