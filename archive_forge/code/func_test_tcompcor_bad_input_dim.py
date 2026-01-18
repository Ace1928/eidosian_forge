import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_tcompcor_bad_input_dim(self):
    bad_dims = (2, 2, 2)
    data_file = utils.save_toy_nii(np.zeros(bad_dims), 'temp.nii')
    interface = TCompCor(realigned_file=data_file)
    with pytest.raises(ValueError):
        interface.run()