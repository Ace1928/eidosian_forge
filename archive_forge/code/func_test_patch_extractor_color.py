import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_patch_extractor_color(orange_face):
    faces = _make_images(orange_face)
    i_h, i_w = faces.shape[1:3]
    p_h, p_w = (8, 8)
    expected_n_patches = len(faces) * (i_h - p_h + 1) * (i_w - p_w + 1)
    extr = PatchExtractor(patch_size=(p_h, p_w), random_state=0)
    patches = extr.transform(faces)
    assert patches.shape == (expected_n_patches, p_h, p_w, 3)