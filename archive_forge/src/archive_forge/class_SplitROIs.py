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
class SplitROIs(BaseInterface):
    """
    Splits a 3D image in small chunks to enable parallel processing.

    ROIs keep time series structure in 4D images.

    Example
    -------
    >>> from nipype.algorithms import misc
    >>> rois = misc.SplitROIs()
    >>> rois.inputs.in_file = 'diffusion.nii'
    >>> rois.inputs.in_mask = 'mask.nii'
    >>> rois.run() # doctest: +SKIP

    """
    input_spec = SplitROIsInputSpec
    output_spec = SplitROIsOutputSpec

    def _run_interface(self, runtime):
        mask = None
        roisize = None
        self._outnames = {}
        if isdefined(self.inputs.in_mask):
            mask = self.inputs.in_mask
        if isdefined(self.inputs.roi_size):
            roisize = self.inputs.roi_size
        res = split_rois(self.inputs.in_file, mask, roisize)
        self._outnames['out_files'] = res[0]
        self._outnames['out_masks'] = res[1]
        self._outnames['out_index'] = res[2]
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for k, v in list(self._outnames.items()):
            outputs[k] = v
        return outputs