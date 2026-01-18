import numpy as np
from skimage import draw
def test_polygon2mask():
    mask = draw.polygon2mask(image_shape, polygon)
    assert mask.shape == image_shape
    assert mask.sum() == 57653