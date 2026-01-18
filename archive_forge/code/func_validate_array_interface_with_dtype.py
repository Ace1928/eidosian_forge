import unittest
import warnings
from io import BytesIO
from itertools import product
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from .. import ecat, minc1, minc2, parrec
from ..analyze import AnalyzeHeader
from ..arrayproxy import ArrayProxy, is_proxy
from ..casting import have_binary128, sctypes
from ..externals.netcdf import netcdf_file
from ..freesurfer.mghformat import MGHHeader
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spm2analyze import Spm2AnalyzeHeader
from ..spm99analyze import Spm99AnalyzeHeader
from ..testing import assert_dt_equal, clear_and_catch_warnings
from ..testing import data_path as DATA_PATH
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import apply_read_scaling
from .test_api_validators import ValidateAPI
from .test_parrec import EG_REC, VARY_REC
def validate_array_interface_with_dtype(self, pmaker, params):
    prox, fio, hdr = pmaker()
    orig = np.array(prox, dtype=None)
    assert_array_equal(orig, params['arr_out'])
    assert_dt_equal(orig.dtype, params['dtype_out'])
    context = None
    if np.issubdtype(orig.dtype, np.complexfloating):
        context = clear_and_catch_warnings()
        context.__enter__()
        warnings.simplefilter('ignore', ComplexWarning)
    for dtype in sctypes['float'] + sctypes['int'] + sctypes['uint']:
        direct = dtype(prox)
        rtol = 0.001 if dtype == np.float16 else 1e-05
        assert_allclose(direct, orig.astype(dtype), rtol=rtol, atol=1e-08)
        assert_dt_equal(direct.dtype, np.dtype(dtype))
        assert direct.shape == params['shape']
        for arrmethod in (np.array, np.asarray, np.asanyarray):
            out = arrmethod(prox, dtype=dtype)
            assert_array_equal(out, direct)
            assert_dt_equal(out.dtype, np.dtype(dtype))
            assert out.shape == params['shape']
    if context is not None:
        context.__exit__()