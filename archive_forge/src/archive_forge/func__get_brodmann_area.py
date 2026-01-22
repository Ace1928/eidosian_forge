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
def _get_brodmann_area(self):
    nii = nb.load(self.inputs.atlas)
    origdata = np.asanyarray(nii.dataobj)
    newdata = np.zeros(origdata.shape)
    if not isinstance(self.inputs.labels, list):
        labels = [self.inputs.labels]
    else:
        labels = self.inputs.labels
    for lab in labels:
        newdata[origdata == lab] = 1
    if self.inputs.hemi == 'right':
        newdata[int(floor(float(origdata.shape[0]) / 2)):, :, :] = 0
    elif self.inputs.hemi == 'left':
        newdata[:int(ceil(float(origdata.shape[0]) / 2)), :, :] = 0
    if self.inputs.dilation_size != 0:
        from scipy.ndimage.morphology import grey_dilation
        newdata = grey_dilation(newdata, (2 * self.inputs.dilation_size + 1, 2 * self.inputs.dilation_size + 1, 2 * self.inputs.dilation_size + 1))
    return nb.Nifti1Image(newdata, nii.affine, nii.header)