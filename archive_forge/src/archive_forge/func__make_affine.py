import os
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import Nifti1Image
from .. import load as top_load
from ..optpkg import optional_package
from .nibabel_data import get_nibabel_data, needs_nibabel_data
def _make_affine(coses, zooms, starts):
    R = np.column_stack(coses)
    Z = np.diag(zooms)
    affine = np.eye(4)
    affine[:3, :3] = np.dot(R, Z)
    affine[:3, 3] = np.dot(R, starts)
    return affine