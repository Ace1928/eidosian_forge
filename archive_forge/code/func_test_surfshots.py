import os
import os.path as op
import pytest
from nipype.testing.fixtures import (
from nipype.pipeline import engine as pe
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import TraitError
from nipype.interfaces.io import FreeSurferSource
@pytest.mark.skipif(fs.no_freesurfer(), reason='freesurfer is not installed')
def test_surfshots(create_files_in_directory_plus_dummy_file):
    fotos = fs.SurfaceSnapshots()
    assert fotos.cmd == 'tksurfer'
    with pytest.raises(ValueError):
        fotos.run()
    files, cwd = create_files_in_directory_plus_dummy_file
    fotos.inputs.subject_id = 'fsaverage'
    fotos.inputs.hemi = 'lh'
    fotos.inputs.surface = 'pial'
    assert fotos.cmdline == 'tksurfer fsaverage lh pial -tcl snapshots.tcl'
    schmotos = fs.SurfaceSnapshots(subject_id='mysubject', hemi='rh', surface='white')
    assert fotos != schmotos
    fotos._write_tcl_script()
    assert os.path.exists('snapshots.tcl')
    foo = open('other.tcl', 'w').close()
    fotos.inputs.tcl_script = 'other.tcl'
    assert fotos.cmdline == 'tksurfer fsaverage lh pial -tcl other.tcl'
    try:
        hold_display = os.environ['DISPLAY']
        del os.environ['DISPLAY']
        with pytest.raises(RuntimeError):
            fotos.run()
        os.environ['DISPLAY'] = hold_display
    except KeyError:
        pass