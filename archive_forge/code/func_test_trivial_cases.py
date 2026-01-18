import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def test_trivial_cases():
    img = np.ones((10, 10))
    labels = np.ones((10, 10))
    with expected_warnings(['Returning provided labels']):
        pass_through = random_walker(img, labels)
    np.testing.assert_array_equal(pass_through, labels)
    labels[:, :5] = 3
    expected = np.concatenate(((labels == 1)[..., np.newaxis], (labels == 3)[..., np.newaxis]), axis=2)
    with expected_warnings(['Returning provided labels']):
        test = random_walker(img, labels, return_full_prob=True)
    np.testing.assert_array_equal(test, expected)
    img = np.full((10, 10), False)
    object_A = np.array([(6, 7), (6, 8), (7, 7), (7, 8)])
    object_B = np.array([(3, 1), (4, 1), (2, 2), (3, 2), (4, 2), (2, 3), (3, 3)])
    for x, y in np.vstack((object_A, object_B)):
        img[y][x] = True
    markers = np.zeros((10, 10), dtype=np.int8)
    for x, y in object_B:
        markers[y][x] = 1
    markers[img == 0] = -1
    with expected_warnings(['All unlabeled pixels are isolated']):
        output_labels = random_walker(img, markers)
    assert np.all(output_labels[markers == 1] == 1)
    assert np.all(output_labels[markers == 0] == -1)
    with expected_warnings(['All unlabeled pixels are isolated']):
        test = random_walker(img, markers, return_full_prob=True)