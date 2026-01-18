import numpy as np
from numpy.testing import assert_equal
from skimage._shared import version_requirements as version_req
from skimage._shared import testing
def test_require():

    @version_req.require('python', '>2.7')
    @version_req.require('numpy', '>1.5')
    def foo():
        return 1
    assert_equal(foo(), 1)

    @version_req.require('scipy', '<0.1')
    def bar():
        return 0
    with testing.raises(ImportError):
        bar()