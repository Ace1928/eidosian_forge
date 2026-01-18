import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_tcompcor_asymmetric_dim(self):
    asymmetric_shape = (2, 3, 4, 5)
    asymmetric_data = utils.save_toy_nii(np.zeros(asymmetric_shape), 'asymmetric.nii')
    TCompCor(realigned_file=asymmetric_data).run()
    assert nb.load('mask_000.nii.gz').shape == asymmetric_shape[:3]