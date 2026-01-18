import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_spatial_filter(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    filter = fsl.SpatialFilter(in_file='a.nii', out_file='b.nii')
    assert filter.cmd == 'fslmaths'
    with pytest.raises(ValueError):
        filter.run()
    for op in ['mean', 'meanu', 'median']:
        filter.inputs.operation = op
        assert filter.cmdline == 'fslmaths a.nii -f{} b.nii'.format(op)
    filter = fsl.SpatialFilter(in_file='a.nii', operation='mean')
    assert filter.cmdline == 'fslmaths a.nii -fmean {}'.format(os.path.join(testdir, 'a_filt{}'.format(out_ext)))