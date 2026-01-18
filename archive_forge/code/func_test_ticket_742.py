import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_ticket_742():

    def SE(img, thresh=0.7, size=4):
        mask = img > thresh
        rank = len(mask.shape)
        la, co = ndimage.label(mask, ndimage.generate_binary_structure(rank, rank))
        _ = ndimage.find_objects(la)
    if np.dtype(np.intp) != np.dtype('i'):
        shape = (3, 1240, 1240)
        a = np.random.rand(np.prod(shape)).reshape(shape)
        SE(a)