import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
def _stimcorr_core(self, motionfile, intensityfile, designmatrix, cwd=None):
    """
        Core routine for determining stimulus correlation

        """
    if not cwd:
        cwd = os.getcwd()
    mc_in = np.loadtxt(motionfile)
    g_in = np.loadtxt(intensityfile)
    g_in.shape = (g_in.shape[0], 1)
    dcol = designmatrix.shape[1]
    mccol = mc_in.shape[1]
    concat_matrix = np.hstack((np.hstack((designmatrix, mc_in)), g_in))
    cm = np.corrcoef(concat_matrix, rowvar=0)
    corrfile = self._get_output_filenames(motionfile, cwd)
    file = open(corrfile, 'w')
    file.write('Stats for:\n')
    file.write('Stimulus correlated motion:\n%s\n' % motionfile)
    for i in range(dcol):
        file.write('SCM.%d:' % i)
        for v in cm[i, dcol + np.arange(mccol)]:
            file.write(' %.2f' % v)
        file.write('\n')
    file.write('Stimulus correlated intensity:\n%s\n' % intensityfile)
    for i in range(dcol):
        file.write('SCI.%d: %.2f\n' % (i, cm[i, -1]))
    file.close()