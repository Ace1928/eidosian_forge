import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage.feature.util import (
@pytest.mark.skipif(not has_mpl, reason='Matplotlib not installed')
@pytest.mark.parametrize('shapes', [((10, 10), (10, 10)), ((10, 10), (12, 10)), ((10, 10), (10, 12)), ((10, 10), (12, 12)), ((12, 10), (10, 10)), ((10, 12), (10, 10)), ((12, 12), (10, 10))])
def test_plot_matches(shapes):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    keypoints1 = 10 * np.random.rand(10, 2)
    keypoints2 = 10 * np.random.rand(10, 2)
    idxs1 = np.random.randint(10, size=10)
    idxs2 = np.random.randint(10, size=10)
    matches = np.column_stack((idxs1, idxs2))
    shape1, shape2 = shapes
    img1 = np.zeros(shape1)
    img2 = np.zeros(shape2)
    with pytest.warns(FutureWarning):
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches)
    with pytest.warns(FutureWarning):
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches, only_matches=True)
    with pytest.warns(FutureWarning):
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches, keypoints_color='r')
    with pytest.warns(FutureWarning):
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches, matches_color='r')
    with pytest.warns(FutureWarning):
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches, alignment='vertical')
    plt.close()