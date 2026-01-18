import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_tcompcor_index_mask(self):
    TCompCor(realigned_file=self.realigned_file, mask_files=self.mask_files, mask_index=1).run()
    assert np.array_equal(nb.load('mask_000.nii.gz').dataobj, [[[0, 0], [0, 0]], [[0, 1], [0, 0]]])