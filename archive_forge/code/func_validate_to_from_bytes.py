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
def validate_to_from_bytes(self, imaker, params):
    img = imaker()
    klass = getattr(self, 'klass', img.__class__)
    with InTemporaryDirectory():
        fname = 'img' + self.standard_extension
        img.to_filename(fname)
        all_images = list(getattr(self, 'example_images', [])) + [{'fname': fname}]
        for img_params in all_images:
            img_a = klass.from_filename(img_params['fname'])
            bytes_a = img_a.to_bytes()
            img_b = klass.from_bytes(bytes_a)
            assert img_b.to_bytes() == bytes_a
            assert self._header_eq(img_a.header, img_b.header)
            assert np.array_equal(img_a.get_fdata(), img_b.get_fdata())
            del img_a
            del img_b