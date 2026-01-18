from numpy.testing import assert_equal
import numpy as np
import skvideo
import skvideo.io
import skvideo.utils
import skvideo.datasets
import os
import nose
def test_vread2x():
    for i in range(2):
        videodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny())