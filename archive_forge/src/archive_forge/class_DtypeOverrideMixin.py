import io
import pathlib
import sys
import warnings
from functools import partial
from itertools import product
import numpy as np
from ..optpkg import optional_package
import unittest
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from nibabel.arraywriters import WriterError
from nibabel.testing import (
from .. import (
from ..casting import sctypes
from ..spatialimages import SpatialImage
from ..tmpdirs import InTemporaryDirectory
from .test_api_validators import ValidateAPI
from .test_brikhead import EXAMPLE_IMAGES as AFNI_EXAMPLE_IMAGES
from .test_minc1 import EXAMPLE_IMAGES as MINC1_EXAMPLE_IMAGES
from .test_minc2 import EXAMPLE_IMAGES as MINC2_EXAMPLE_IMAGES
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLE_IMAGES
class DtypeOverrideMixin(GetSetDtypeMixin):
    """Test images that can accept ``dtype`` arguments to ``__init__`` and
    ``to_file_map``
    """

    def validate_init_dtype_override(self, imaker, params):
        img = imaker()
        klass = img.__class__
        for dtype in self.storable_dtypes:
            if hasattr(img, 'affine'):
                new_img = klass(img.dataobj, img.affine, header=img.header, dtype=dtype)
            else:
                new_img = klass(img.dataobj, header=img.header, dtype=dtype)
            assert new_img.get_data_dtype() == dtype
            if self.has_scaling and self.can_save:
                with np.errstate(invalid='ignore'):
                    rt_img = bytesio_round_trip(new_img)
                assert rt_img.get_data_dtype() == dtype

    def validate_to_file_dtype_override(self, imaker, params):
        if not self.can_save:
            raise unittest.SkipTest
        img = imaker()
        orig_dtype = img.get_data_dtype()
        fname = 'image' + self.standard_extension
        with InTemporaryDirectory():
            for dtype in self.storable_dtypes:
                try:
                    img.to_filename(fname, dtype=dtype)
                except WriterError:
                    continue
                rt_img = img.__class__.from_filename(fname)
                assert rt_img.get_data_dtype() == dtype
                assert img.get_data_dtype() == orig_dtype