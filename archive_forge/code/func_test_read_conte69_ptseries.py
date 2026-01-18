import os
import tempfile
import numpy as np
import nibabel as nib
from nibabel.cifti2 import cifti2, cifti2_axes
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_ptseries():
    img = nib.load(os.path.join(test_directory, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.ptseries.nii'))
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert isinstance(axes[0], cifti2_axes.SeriesAxis)
    assert len(axes[0]) == 2
    assert axes[0].start == 0
    assert axes[0].step == 1
    assert axes[0].size == arr.shape[0]
    assert (axes[0].time == [0, 1]).all()
    assert len(axes[1]) == 54
    voxels, vertices = axes[1]['ER_FRB08']
    assert voxels.shape == (0, 3)
    assert len(vertices) == 2
    assert vertices['CIFTI_STRUCTURE_CORTEX_LEFT'].shape == (206 // 2,)
    assert vertices['CIFTI_STRUCTURE_CORTEX_RIGHT'].shape == (206 // 2,)
    check_rewrite(arr, axes)