import os
import numpy as np
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm.base as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.interfaces.base import traits
def test_reformat_dict_for_savemat():

    class TestClass(spm.SPMCommand):
        input_spec = spm.SPMCommandInputSpec
    dc = TestClass()
    out = dc._reformat_dict_for_savemat({'a': {'b': {'c': []}}})
    assert out == [{'a': [{'b': [{'c': []}]}]}]