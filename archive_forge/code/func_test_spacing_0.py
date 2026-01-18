import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def test_spacing_0():
    n = 30
    lx, ly, lz = (n, n, n)
    data, _ = make_3d_syntheticdata(lx, ly, lz)
    data_aniso = np.zeros((n, n, n // 2))
    for i, yz in enumerate(data):
        data_aniso[i, :, :] = resize(yz, (n, n // 2), mode='constant', anti_aliasing=False)
    small_l = int(lx // 5)
    labels_aniso = np.zeros_like(data_aniso)
    labels_aniso[lx // 5, ly // 5, lz // 5] = 1
    labels_aniso[lx // 2 + small_l // 4, ly // 2 - small_l // 4, lz // 4 - small_l // 8] = 2
    with expected_warnings(['"cg" mode|scipy.sparse.linalg.cg']):
        labels_aniso = random_walker(data_aniso, labels_aniso, mode='cg', spacing=(1.0, 1.0, 0.5))
    assert (labels_aniso[13:17, 13:17, 7:9] == 2).all()