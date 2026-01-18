import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_realign_list_outputs(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    rlgn = spm.Realign(in_files=filelist[0])
    assert rlgn._list_outputs()['realignment_parameters'][0].startswith('rp_')
    assert rlgn._list_outputs()['realigned_files'][0].startswith('r')
    assert rlgn._list_outputs()['mean_image'].startswith('mean')