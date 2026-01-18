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
def validate_is_proxy(self, pmaker, params):
    prox, fio, hdr = pmaker()
    assert prox.is_proxy
    assert is_proxy(prox)
    assert not is_proxy(np.arange(10))
    with pytest.raises(AttributeError):
        prox.is_proxy = False