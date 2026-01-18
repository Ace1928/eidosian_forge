import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_projective_repr():
    tform = ProjectiveTransform()
    want = re.escape(textwrap.dedent('\n        <ProjectiveTransform(matrix=\n            [[1., 0., 0.],\n             [0., 1., 0.],\n             [0., 0., 1.]]) at\n        ').strip()) + ' 0x[a-f0-9]+' + re.escape('>')
    want = want.replace('0\\.', ' *0\\.')
    want = want.replace('1\\.', ' *1\\.')
    assert re.match(want, repr(tform))