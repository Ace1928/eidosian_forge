import pytest
import numpy as np
from h5py import File
from .common import TestCase
from .data_files import get_data_file_path
def test_vlen_big_endian():
    with File(get_data_file_path('vlen_string_s390x.h5')) as f:
        assert f.attrs['created_on_s390x'] == 1
        dset = f['DSvariable']
        assert dset[0] == b'Parting'
        assert dset[1] == b'is such'
        assert dset[2] == b'sweet'
        assert dset[3] == b'sorrow...'
        dset = f['DSLEfloat']
        assert dset[0] == 3.14
        assert dset[1] == 1.61
        assert dset[2] == 2.71
        assert dset[3] == 2.41
        assert dset[4] == 1.2
        assert dset.dtype == '<f8'
        assert f['DSBEfloat'][0] == 3.14
        assert f['DSBEfloat'].dtype == '>f8'
        assert f['DSLEint'][0] == 1
        assert f['DSLEint'].dtype == '<u8'
        assert f['DSBEint'][0] == 1
        assert f['DSBEint'].dtype == '>i8'