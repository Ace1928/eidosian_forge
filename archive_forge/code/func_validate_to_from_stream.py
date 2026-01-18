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
def validate_to_from_stream(self, imaker, params):
    img = imaker()
    klass = getattr(self, 'klass', img.__class__)
    stream = io.BytesIO()
    img.to_stream(stream)
    rt_img = klass.from_stream(stream)
    assert self._header_eq(img.header, rt_img.header)
    assert np.array_equal(img.get_fdata(), rt_img.get_fdata())