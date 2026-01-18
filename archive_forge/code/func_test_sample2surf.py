import os
import os.path as op
import pytest
from nipype.testing.fixtures import (
from nipype.pipeline import engine as pe
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import TraitError
from nipype.interfaces.io import FreeSurferSource
@pytest.mark.skipif(fs.no_freesurfer(), reason='freesurfer is not installed')
def test_sample2surf(create_files_in_directory_plus_dummy_file):
    s2s = fs.SampleToSurface()
    assert s2s.cmd == 'mri_vol2surf'
    with pytest.raises(ValueError):
        s2s.run()
    files, cwd = create_files_in_directory_plus_dummy_file
    s2s.inputs.source_file = files[0]
    s2s.inputs.reference_file = files[1]
    s2s.inputs.hemi = 'lh'
    s2s.inputs.reg_file = files[2]
    s2s.inputs.sampling_range = 0.5
    s2s.inputs.sampling_units = 'frac'
    s2s.inputs.sampling_method = 'point'
    assert s2s.cmdline == 'mri_vol2surf --hemi lh --o %s --ref %s --reg reg.dat --projfrac 0.500 --mov %s' % (os.path.join(cwd, 'lh.a.mgz'), files[1], files[0])
    s2sish = fs.SampleToSurface(source_file=files[1], reference_file=files[0], hemi='rh')
    assert s2s != s2sish
    s2s.inputs.hits_file = True
    assert s2s._get_outfilename('hits_file') == os.path.join(cwd, 'lh.a_hits.mgz')

    def set_illegal_range():
        s2s.inputs.sampling_range = (0.2, 0.5)
    with pytest.raises(TraitError):
        set_illegal_range()