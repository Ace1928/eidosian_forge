import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_normalize_list_outputs(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    norm = spm.Normalize(source=filelist[0])
    assert norm._list_outputs()['normalized_source'][0].startswith('w')
    norm = spm.Normalize(source=filelist[0], apply_to_files=filelist[1])
    assert norm._list_outputs()['normalized_files'][0].startswith('w')