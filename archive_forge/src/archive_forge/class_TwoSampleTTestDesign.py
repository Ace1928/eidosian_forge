import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class TwoSampleTTestDesign(FactorialDesign):
    """Create SPM design for two sample t-test

    Examples
    --------

    >>> ttest = TwoSampleTTestDesign()
    >>> ttest.inputs.group1_files = ['cont1.nii', 'cont2.nii']
    >>> ttest.inputs.group2_files = ['cont1a.nii', 'cont2a.nii']
    >>> ttest.run() # doctest: +SKIP
    """
    input_spec = TwoSampleTTestDesignInputSpec

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['group1_files', 'group2_files']:
            return np.array(val, dtype=object)
        return super(TwoSampleTTestDesign, self)._format_arg(opt, spec, val)