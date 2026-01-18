import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_normalize12_list_outputs(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    norm12 = spm.Normalize12(image_to_align=filelist[0])
    assert norm12._list_outputs()['normalized_image'][0].startswith('w')
    norm12 = spm.Normalize12(image_to_align=filelist[0], apply_to_files=filelist[1])
    assert norm12._list_outputs()['normalized_files'][0].startswith('w')