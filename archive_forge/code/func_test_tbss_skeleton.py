import os
import nipype.interfaces.fsl.dti as fsl
from nipype.interfaces.fsl import Info, no_fsl
from nipype.interfaces.base import Undefined
import pytest
from nipype.testing.fixtures import create_files_in_directory
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_tbss_skeleton(create_files_in_directory):
    skeletor = fsl.TractSkeleton()
    files, newdir = create_files_in_directory
    assert skeletor.cmd == 'tbss_skeleton'
    with pytest.raises(ValueError):
        skeletor.run()
    skeletor.inputs.in_file = files[0]
    skeletor.inputs.skeleton_file = True
    assert skeletor.cmdline == 'tbss_skeleton -i a.nii -o %s' % os.path.join(newdir, 'a_skeleton.nii')
    skeletor.inputs.skeleton_file = 'old_boney.nii'
    assert skeletor.cmdline == 'tbss_skeleton -i a.nii -o old_boney.nii'
    bones = fsl.TractSkeleton(in_file='a.nii', project_data=True)
    with pytest.raises(ValueError):
        bones.run()
    bones.inputs.threshold = 0.2
    bones.inputs.distance_map = 'b.nii'
    bones.inputs.data_file = 'b.nii'
    assert bones.cmdline == 'tbss_skeleton -i a.nii -p 0.200 b.nii %s b.nii %s' % (Info.standard_image('LowerCingulum_1mm.nii.gz'), os.path.join(newdir, 'b_skeletonised.nii'))
    bones.inputs.use_cingulum_mask = Undefined
    bones.inputs.search_mask_file = 'a.nii'
    assert bones.cmdline == 'tbss_skeleton -i a.nii -p 0.200 b.nii a.nii b.nii %s' % os.path.join(newdir, 'b_skeletonised.nii')