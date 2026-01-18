import os
import os.path as op
import pytest
from nipype.testing.fixtures import (
from nipype.pipeline import engine as pe
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import TraitError
from nipype.interfaces.io import FreeSurferSource
@pytest.mark.skipif(fs.no_freesurfer(), reason='freesurfer is not installed')
def test_surfxfm(create_surf_file_in_directory):
    xfm = fs.SurfaceTransform()
    assert xfm.cmd == 'mri_surf2surf'
    with pytest.raises(ValueError):
        xfm.run()
    surf, cwd = create_surf_file_in_directory
    xfm.inputs.source_file = surf
    xfm.inputs.source_subject = 'my_subject'
    xfm.inputs.target_subject = 'fsaverage'
    xfm.inputs.hemi = 'lh'
    assert xfm.cmdline == 'mri_surf2surf --hemi lh --tval %s/lh.a.fsaverage.nii --sval %s --srcsubject my_subject --trgsubject fsaverage' % (cwd, surf)
    xfmish = fs.SurfaceTransform(source_subject='fsaverage', target_subject='my_subject', source_file=surf, hemi='lh')
    assert xfm != xfmish