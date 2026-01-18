import os
import numpy as np
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm.base as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.interfaces.base import traits
def test_scan_for_fnames(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    names = spm.scans_for_fnames(filelist, keep4d=True)
    assert names[0] == filelist[0]
    assert names[1] == filelist[1]