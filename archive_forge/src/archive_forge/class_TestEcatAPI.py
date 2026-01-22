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
class TestEcatAPI(_TestProxyAPI):
    eg_fname = 'tinypet.v'
    eg_shape = (10, 10, 3, 1)

    def obj_params(self):
        eg_path = pjoin(DATA_PATH, self.eg_fname)
        img = ecat.load(eg_path)
        arr_out = img.get_fdata()

        def eg_func():
            img = ecat.load(eg_path)
            sh = img.get_subheaders()
            prox = ecat.EcatImageArrayProxy(sh)
            fobj = open(eg_path, 'rb')
            return (prox, fobj, sh)
        yield (eg_func, dict(shape=self.eg_shape, dtype_out=np.float64, arr_out=arr_out))

    def validate_header_isolated(self, pmaker, params):
        raise unittest.SkipTest('ECAT header does not support dtype get')