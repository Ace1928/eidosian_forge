import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def _make_images(face):
    images = np.zeros((3,) + face.shape)
    images[0] = face
    images[1] = face + 1
    images[2] = face + 2
    return images