import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
def test_none_affine(self):
    img_klass = self.image_class
    img = img_klass(np.zeros((2, 3, 4)), None)
    aff = img.header.get_best_affine()
    for key, value in img.file_map.items():
        value.fileobj = BytesIO()
    img.to_file_map()
    img_back = img.from_file_map(img.file_map)
    assert_array_equal(img_back.affine, aff)