import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_unarymaths(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    maths = fsl.UnaryMaths(in_file='a.nii', out_file='b.nii')
    assert maths.cmd == 'fslmaths'
    with pytest.raises(ValueError):
        maths.run()
    ops = ['exp', 'log', 'sin', 'cos', 'sqr', 'sqrt', 'recip', 'abs', 'bin', 'index']
    for op in ops:
        maths.inputs.operation = op
        assert maths.cmdline == 'fslmaths a.nii -{} b.nii'.format(op)
    for op in ops:
        maths = fsl.UnaryMaths(in_file='a.nii', operation=op)
        assert maths.cmdline == 'fslmaths a.nii -{} {}'.format(op, os.path.join(testdir, 'a_{}{}'.format(op, out_ext)))