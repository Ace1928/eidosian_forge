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
@needs_nibabel_data('nitest-cifti2')
def test_read_scalar():
    img = ci.Cifti2Image.from_filename(DATA_FILE2)
    scalar_mapping = img.header.matrix.get_index_map(0)
    expected_names = ('MyelinMap_BC_decurv', 'corrThickness')
    assert img.shape[0] == len(expected_names)
    assert len(list(scalar_mapping.named_maps)) == len(expected_names)
    expected_meta = [('PaletteColorMapping', '<PaletteColorMapping Version="1">\n   <ScaleMo')]
    for scalar, name in zip(scalar_mapping.named_maps, expected_names):
        assert scalar.map_name == name
        assert len(scalar.metadata) == len(expected_meta)
        print(expected_meta[0], scalar.metadata.data.keys())
        for key, value in expected_meta:
            assert key in scalar.metadata.data.keys()
            assert scalar.metadata[key][:len(value)] == value
        assert scalar.label_table is None, '.dscalar file should not define a label table'