import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def test_rag_error():
    img = np.zeros((10, 10, 3), dtype='uint8')
    labels = np.zeros((10, 10), dtype='uint8')
    labels[:5, :] = 0
    labels[5:, :] = 1
    with testing.raises(ValueError):
        graph.rag_mean_color(img, labels, 2, 'non existent mode')