import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MotionOutliers(FSLCommand):
    """
    Use FSL fsl_motion_outliers`http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLMotionOutliers`_ to find outliers in timeseries (4d) data.
    Examples
    --------
    >>> from nipype.interfaces.fsl import MotionOutliers
    >>> mo = MotionOutliers()
    >>> mo.inputs.in_file = "epi.nii"
    >>> mo.cmdline # doctest: +ELLIPSIS
    'fsl_motion_outliers -i epi.nii -o epi_outliers.txt -p epi_metrics.png -s epi_metrics.txt'
    >>> res = mo.run() # doctest: +SKIP
    """
    input_spec = MotionOutliersInputSpec
    output_spec = MotionOutliersOutputSpec
    _cmd = 'fsl_motion_outliers'