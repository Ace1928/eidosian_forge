import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_slicetiming_list_outputs(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    st = spm.SliceTiming(in_files=filelist[0])
    assert st._list_outputs()['timecorrected_files'][0][0] == 'a'