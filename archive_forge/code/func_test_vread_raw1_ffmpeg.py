from numpy.testing import assert_equal
import numpy as np
import skvideo
import skvideo.io
import skvideo.utils
import skvideo.datasets
import os
import nose
def test_vread_raw1_ffmpeg():
    _rawhelper1('ffmpeg')