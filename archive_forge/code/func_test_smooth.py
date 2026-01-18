import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_smooth():
    assert spm.Smooth._jobtype == 'spatial'
    assert spm.Smooth._jobname == 'smooth'