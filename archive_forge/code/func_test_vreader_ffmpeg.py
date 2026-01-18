from numpy.testing import assert_equal
import numpy as np
import skvideo.io
import skvideo.datasets
import sys
@unittest.skipIf(not skvideo._HAS_FFMPEG, 'FFmpeg required for this test.')
def test_vreader_ffmpeg():
    _vreader('ffmpeg')