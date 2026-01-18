import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_slicetiming():
    assert spm.SliceTiming._jobtype == 'temporal'
    assert spm.SliceTiming._jobname == 'st'