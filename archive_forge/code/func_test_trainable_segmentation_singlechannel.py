from functools import partial
import numpy as np
import pytest
from scipy import spatial
from skimage.future import fit_segmenter, predict_segmenter, TrainableSegmenter
from skimage.feature import multiscale_basic_features
def test_trainable_segmentation_singlechannel():
    img = np.zeros((20, 20))
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    labels = np.zeros_like(img, dtype=np.uint8)
    labels[:2] = 1
    labels[-2:] = 2
    clf = DummyNNClassifier()
    features_func = partial(multiscale_basic_features, edges=False, texture=False, sigma_min=0.5, sigma_max=2)
    features = features_func(img)
    clf = fit_segmenter(labels, features, clf)
    out = predict_segmenter(features, clf)
    assert np.all(out[:10] == 1)
    assert np.all(out[10:] == 2)