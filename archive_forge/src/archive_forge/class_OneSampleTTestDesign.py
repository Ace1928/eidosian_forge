import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class OneSampleTTestDesign(FactorialDesign):
    """Create SPM design for one sample t-test

    Examples
    --------

    >>> ttest = OneSampleTTestDesign()
    >>> ttest.inputs.in_files = ['cont1.nii', 'cont2.nii']
    >>> ttest.run() # doctest: +SKIP
    """
    input_spec = OneSampleTTestDesignInputSpec

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['in_files']:
            return np.array(val, dtype=object)
        return super(OneSampleTTestDesign, self)._format_arg(opt, spec, val)