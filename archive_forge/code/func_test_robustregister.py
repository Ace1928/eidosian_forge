import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
from nipype.interfaces import freesurfer
from nipype.interfaces.freesurfer import Info
from nipype import LooseVersion
@pytest.mark.skipif(freesurfer.no_freesurfer(), reason='freesurfer is not installed')
def test_robustregister(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    reg = freesurfer.RobustRegister()
    cwd = os.getcwd()
    assert reg.cmd == 'mri_robust_register'
    with pytest.raises(ValueError):
        reg.run()
    reg.inputs.source_file = filelist[0]
    reg.inputs.target_file = filelist[1]
    reg.inputs.auto_sens = True
    assert reg.cmdline == 'mri_robust_register --satit --lta %s/%s_robustreg.lta --mov %s --dst %s' % (cwd, filelist[0][:-4], filelist[0], filelist[1])
    reg2 = freesurfer.RobustRegister(source_file=filelist[0], target_file=filelist[1], outlier_sens=3.0, out_reg_file='foo.lta', half_targ=True)
    assert reg2.cmdline == 'mri_robust_register --halfdst %s_halfway.nii --lta foo.lta --sat 3.0000 --mov %s --dst %s' % (os.path.join(outdir, filelist[1][:-4]), filelist[0], filelist[1])