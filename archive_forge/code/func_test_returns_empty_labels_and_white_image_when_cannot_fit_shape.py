import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_returns_empty_labels_and_white_image_when_cannot_fit_shape():
    with expected_warnings(['Could not fit']):
        image, labels = random_shapes((10000, 10000), max_shapes=1, min_size=10000, shape='circle')
    assert len(labels) == 0
    assert (image == 255).all()