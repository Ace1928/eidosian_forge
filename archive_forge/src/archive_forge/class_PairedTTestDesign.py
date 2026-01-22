import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class PairedTTestDesign(FactorialDesign):
    """Create SPM design for paired t-test

    Examples
    --------

    >>> pttest = PairedTTestDesign()
    >>> pttest.inputs.paired_files = [['cont1.nii','cont1a.nii'],['cont2.nii','cont2a.nii']]
    >>> pttest.run() # doctest: +SKIP
    """
    input_spec = PairedTTestDesignInputSpec

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['paired_files']:
            return [dict(scans=np.array(files, dtype=object)) for files in val]
        return super(PairedTTestDesign, self)._format_arg(opt, spec, val)