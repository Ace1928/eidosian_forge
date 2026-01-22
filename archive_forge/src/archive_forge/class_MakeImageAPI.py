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
class MakeImageAPI(LoadImageAPI):
    """Validation for images we can make with ``func(data, affine, header)``"""
    image_maker = None
    header_maker = None
    example_shapes = ((2,), (2, 3), (2, 3, 4), (2, 3, 4, 5))
    storable_dtypes = (np.uint8, np.int16, np.float32)

    def obj_params(self):
        for func, params in super().obj_params():
            yield (func, params)
        aff = np.diag([1, 2, 3, 1])

        def make_imaker(arr, aff, header=None):
            return lambda: self.image_maker(arr, aff, header)

        def make_prox_imaker(arr, aff, hdr):

            def prox_imaker():
                img = self.image_maker(arr, aff, hdr)
                rt_img = bytesio_round_trip(img)
                return self.image_maker(rt_img.dataobj, aff, rt_img.header)
            return prox_imaker
        for shape, stored_dtype in product(self.example_shapes, self.storable_dtypes):
            arr = np.arange(np.prod(shape), dtype=stored_dtype).reshape(shape)
            hdr = self.header_maker()
            hdr.set_data_dtype(stored_dtype)
            func = make_imaker(arr.copy(), aff, hdr)
            params = dict(dtype=stored_dtype, affine=aff, data=arr, shape=shape, is_proxy=False)
            yield (make_imaker(arr.copy(), aff, hdr), params)
            if not self.can_save:
                continue
            params['is_proxy'] = True
            yield (make_prox_imaker(arr.copy(), aff, hdr), params)