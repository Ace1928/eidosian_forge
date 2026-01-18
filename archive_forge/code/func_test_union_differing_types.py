import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_union_differing_types():
    tform1 = SimilarityTransform()
    tform2 = PolynomialTransform()
    with pytest.raises(TypeError):
        tform1.__add__(tform2)