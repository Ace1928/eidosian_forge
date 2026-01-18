from numpy.testing import assert_equal
import numpy as np
import skvideo
import skvideo.io
import skvideo.utils
import skvideo.datasets
import os
import nose
def test_vread():
    videodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
    T, M, N, C = videodata.shape
    assert_equal(T, 132)
    assert_equal(M, 720)
    assert_equal(N, 1280)
    assert_equal(C, 3)
    assert_equal(np.mean(videodata), 109.28332841215979)