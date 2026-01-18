import os
import numpy as np
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm.base as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.interfaces.base import traits
@pytest.mark.skipif(no_spm(), reason='spm is not installed')
def test_spm_path():
    spm_path = spm.Info.path()
    if spm_path is not None:
        assert isinstance(spm_path, (str, bytes))
        assert 'spm' in spm_path.lower()