import io
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from packaging.version import Version
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader
from nibabel.tests import test_nifti2 as tn2
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tmpdirs import InTemporaryDirectory
def test_read_nifti2():
    filemap = ci.Cifti2Image.make_file_map()
    for k in filemap:
        filemap[k].fileobj = open(NIFTI2_DATA)
    with pytest.raises(ValueError):
        ci.Cifti2Image.from_file_map(filemap)