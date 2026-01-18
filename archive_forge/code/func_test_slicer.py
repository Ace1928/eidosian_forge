import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_slicer(create_files_in_directory_plus_output_type):
    filelist, outdir, _ = create_files_in_directory_plus_output_type
    slicer = fsl.Slicer()
    assert slicer.cmd == 'slicer'
    with pytest.raises(ValueError):
        slicer.run()
    slicer.inputs.in_file = filelist[0]
    slicer.inputs.image_edges = filelist[1]
    slicer.inputs.intensity_range = (10.0, 20.0)
    slicer.inputs.all_axial = True
    slicer.inputs.image_width = 750
    slicer.inputs.out_file = 'foo_bar.png'
    assert slicer.cmdline == 'slicer %s %s -L -i 10.000 20.000  -A 750 foo_bar.png' % (filelist[0], filelist[1])
    slicer2 = fsl.Slicer(in_file=filelist[0], middle_slices=True, label_slices=False, out_file='foo_bar2.png')
    assert slicer2.cmdline == 'slicer %s   -a foo_bar2.png' % filelist[0]