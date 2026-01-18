import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_realign():
    assert spm.Realign._jobtype == 'spatial'
    assert spm.Realign._jobname == 'realign'
    assert spm.Realign().inputs.jobtype == 'estwrite'