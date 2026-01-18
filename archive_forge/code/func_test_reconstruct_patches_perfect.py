import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_reconstruct_patches_perfect(downsampled_face):
    face = downsampled_face
    p_h, p_w = (16, 16)
    patches = extract_patches_2d(face, (p_h, p_w))
    face_reconstructed = reconstruct_from_patches_2d(patches, face.shape)
    np.testing.assert_array_almost_equal(face, face_reconstructed)