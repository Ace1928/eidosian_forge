import os
import numpy as np
import nibabel as nb
from ..interfaces.base import (
def _4d(self, array, affine):
    """takes a 3-dimensional numpy array and an affine,
        returns the equivalent 4th dimensional nifti file"""
    return nb.Nifti1Image(array[:, :, :, np.newaxis], affine)