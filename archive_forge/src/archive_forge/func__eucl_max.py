import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
def _eucl_max(self, nii1, nii2):
    from scipy.spatial.distance import cdist
    origdata1 = np.asanyarray(nii1.dataobj)
    origdata1 = (np.rint(origdata1) != 0) & ~np.isnan(origdata1)
    origdata2 = np.asanyarray(nii2.dataobj)
    origdata2 = (np.rint(origdata2) != 0) & ~np.isnan(origdata2)
    if isdefined(self.inputs.mask_volume):
        maskdata = np.asanyarray(nb.load(self.inputs.mask_volume).dataobj)
        maskdata = (np.rint(maskdata) != 0) & ~np.isnan(maskdata)
        origdata1 = np.logical_and(maskdata, origdata1)
        origdata2 = np.logical_and(maskdata, origdata2)
    if origdata1.max() == 0 or origdata2.max() == 0:
        return np.nan
    border1 = self._find_border(origdata1)
    border2 = self._find_border(origdata2)
    set1_coordinates = self._get_coordinates(border1, nii1.affine)
    set2_coordinates = self._get_coordinates(border2, nii2.affine)
    distances = cdist(set1_coordinates.T, set2_coordinates.T)
    mins = np.concatenate((np.amin(distances, axis=0), np.amin(distances, axis=1)))
    return np.max(mins)