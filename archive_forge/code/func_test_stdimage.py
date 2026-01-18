import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_stdimage(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    stder = fsl.StdImage(in_file='a.nii', out_file='b.nii')
    assert stder.cmd == 'fslmaths'
    assert stder.cmdline == 'fslmaths a.nii -Tstd b.nii'
    cmdline = 'fslmaths a.nii -{}std b.nii'
    for dim in ['X', 'Y', 'Z', 'T']:
        stder.inputs.dimension = dim
        assert stder.cmdline == cmdline.format(dim)
    stder = fsl.StdImage(in_file='a.nii', output_type='NIFTI')
    assert stder.cmdline == 'fslmaths a.nii -Tstd {}'.format(os.path.join(testdir, 'a_std.nii'))