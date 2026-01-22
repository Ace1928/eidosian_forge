from numpy.testing import assert_equal
import numpy as np
import skvideo
import skvideo.io
import skvideo.utils
import skvideo.datasets
import os
import nose
def _rawhelper2(backend):
    pipingDict = {'-pix_fmt': 'yuvj444p'}
    bunnyMP4VideoData1 = skvideo.io.vread(skvideo.datasets.bigbuckbunny(), num_frames=1, outputdict=pipingDict.copy(), backend=backend)
    skvideo.io.vwrite('bunnyMP4VideoData_vwrite.yuv', bunnyMP4VideoData1, inputdict=pipingDict.copy(), backend=backend)
    bunnyYUVVideoData1 = skvideo.io.vread('bunnyMP4VideoData_vwrite.yuv', width=1280, height=720, num_frames=1, outputdict=pipingDict.copy(), backend=backend)
    skvideo.io.vwrite('bunnyYUVVideoData_vwrite.yuv', bunnyYUVVideoData1, inputdict=pipingDict.copy(), backend=backend)
    bunnyYUVVideoData2 = skvideo.io.vread('bunnyYUVVideoData_vwrite.yuv', width=1280, height=720, num_frames=1, outputdict=pipingDict.copy(), backend=backend)
    bunnyMP4VideoData2 = skvideo.io.vread(skvideo.datasets.bigbuckbunny(), num_frames=1, outputdict=pipingDict.copy(), backend=backend)
    assert_equal(bunnyMP4VideoData1.shape, (1, 720, 1280, 3))
    assert_equal(bunnyMP4VideoData2.shape, (1, 720, 1280, 3))
    assert_equal(bunnyYUVVideoData1.shape, (1, 720, 1280, 3))
    assert_equal(bunnyYUVVideoData2.shape, (1, 720, 1280, 3))
    t = np.mean((bunnyMP4VideoData1 - bunnyMP4VideoData2) ** 2)
    assert t == 0, 'Possible mutable default error in vread. MSE=%f between consecutive reads.' % (t,)
    t = np.mean((bunnyMP4VideoData1 - bunnyYUVVideoData1) ** 2)
    assert t == 0, 'Precision loss (mse=%f) performing vwrite (mp4 data) -> vread (raw data).' % (t,)
    t = np.mean((bunnyYUVVideoData1 - bunnyYUVVideoData2) ** 2)
    assert t == 0, 'Unacceptable precision loss (mse=%f) performing vwrite (raw data) -> vread (raw data).' % (t,)
    os.remove('bunnyMP4VideoData_vwrite.yuv')
    os.remove('bunnyYUVVideoData_vwrite.yuv')