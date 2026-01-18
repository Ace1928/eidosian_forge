import numpy as np
import pytest
from skimage import io
from skimage._shared._warnings import expected_warnings
def test_uint8():
    plt.figure()
    with expected_warnings(imshow_expected_warnings + ['CObject type is marked|\\A\\Z']):
        ax_im = io.imshow(im8)
    assert ax_im.cmap.name == 'gray'
    assert ax_im.get_clim() == (0, 255)
    assert n_subplots(ax_im) == 1
    assert ax_im.colorbar is None