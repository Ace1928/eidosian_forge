import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
@pytest.mark.skipif(np.iinfo('l').max < 2 ** 32, reason='Cannot test with 32-bit C long')
def test_repeatability_32bit_boundary_broadcasting(self):
    desired = np.array([[[3992670689, 2438360420, 2557845020], [4107320065, 4142558326, 3216529513], [1605979228, 2807061240, 665605495]], [[3211410639, 4128781000, 457175120], [1712592594, 1282922662, 3081439808], [3997822960, 2008322436, 1563495165]], [[1398375547, 4269260146, 115316740], [3414372578, 3437564012, 2112038651], [3572980305, 2260248732, 3908238631]], [[2561372503, 223155946, 3127879445], [441282060, 3514786552, 2148440361], [1629275283, 3479737011, 3003195987]], [[412181688, 940383289, 3047321305], [2978368172, 764731833, 2282559898], [105711276, 720447391, 3596512484]]])
    for size in [None, (5, 3, 3)]:
        random.seed(12345)
        x = self.rfunc([[-1], [0], [1]], [2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1], size=size)
        assert_array_equal(x, desired if size is not None else desired[0])