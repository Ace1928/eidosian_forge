import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
def _eucl_mean(self, nii1, nii2, weighted=False):
    from scipy.spatial.distance import cdist
    origdata1 = np.asanyarray(nii1.dataobj).astype(bool)
    border1 = self._find_border(origdata1)
    origdata2 = np.asanyarray(nii2.dataobj).astype(bool)
    set1_coordinates = self._get_coordinates(border1, nii1.affine)
    set2_coordinates = self._get_coordinates(origdata2, nii2.affine)
    dist_matrix = cdist(set1_coordinates.T, set2_coordinates.T)
    min_dist_matrix = np.amin(dist_matrix, axis=0)
    import matplotlib
    matplotlib.use(config.get('execution', 'matplotlib_backend'))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(min_dist_matrix, 50, density=True, facecolor='green')
    plt.savefig(self._hist_filename)
    plt.clf()
    plt.close()
    if weighted:
        return np.average(min_dist_matrix, weights=nii2.dataobj[origdata2].flat)
    else:
        return np.mean(min_dist_matrix)