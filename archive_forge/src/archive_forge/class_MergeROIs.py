import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class MergeROIs(BaseInterface):
    """
    Splits a 3D image in small chunks to enable parallel processing.

    ROIs keep time series structure in 4D images.

    Example
    -------
    >>> from nipype.algorithms import misc
    >>> rois = misc.MergeROIs()
    >>> rois.inputs.in_files = ['roi%02d.nii' % i for i in range(1, 6)]
    >>> rois.inputs.in_reference = 'mask.nii'
    >>> rois.inputs.in_index = ['roi%02d_idx.npz' % i for i in range(1, 6)]
    >>> rois.run() # doctest: +SKIP

    """
    input_spec = MergeROIsInputSpec
    output_spec = MergeROIsOutputSpec

    def _run_interface(self, runtime):
        res = merge_rois(self.inputs.in_files, self.inputs.in_index, self.inputs.in_reference)
        self._merged = res
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['merged_file'] = self._merged
        return outputs