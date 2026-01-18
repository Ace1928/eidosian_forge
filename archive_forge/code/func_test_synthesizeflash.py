import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
from nipype.interfaces import freesurfer
from nipype.interfaces.freesurfer import Info
from nipype import LooseVersion
@pytest.mark.skipif(freesurfer.no_freesurfer(), reason='freesurfer is not installed')
def test_synthesizeflash(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    syn = freesurfer.SynthesizeFLASH()
    assert syn.cmd == 'mri_synthesize'
    with pytest.raises(ValueError):
        syn.run()
    syn.inputs.t1_image = filelist[0]
    syn.inputs.pd_image = filelist[1]
    syn.inputs.flip_angle = 30
    syn.inputs.te = 4.5
    syn.inputs.tr = 20
    assert syn.cmdline == 'mri_synthesize 20.00 30.00 4.500 %s %s %s' % (filelist[0], filelist[1], os.path.join(outdir, 'synth-flash_30.mgz'))
    syn2 = freesurfer.SynthesizeFLASH(t1_image=filelist[0], pd_image=filelist[1], flip_angle=20, te=5, tr=25)
    assert syn2.cmdline == 'mri_synthesize 25.00 20.00 5.000 %s %s %s' % (filelist[0], filelist[1], os.path.join(outdir, 'synth-flash_20.mgz'))