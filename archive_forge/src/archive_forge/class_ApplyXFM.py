import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ApplyXFM(FLIRT):
    """Currently just a light wrapper around FLIRT,
    with no modifications

    ApplyXFM is used to apply an existing transform to an image


    Examples
    --------

    >>> import nipype.interfaces.fsl as fsl
    >>> from nipype.testing import example_data
    >>> applyxfm = fsl.preprocess.ApplyXFM()
    >>> applyxfm.inputs.in_file = example_data('structural.nii')
    >>> applyxfm.inputs.in_matrix_file = example_data('trans.mat')
    >>> applyxfm.inputs.out_file = 'newfile.nii'
    >>> applyxfm.inputs.reference = example_data('mni.nii')
    >>> applyxfm.inputs.apply_xfm = True
    >>> result = applyxfm.run() # doctest: +SKIP

    """
    input_spec = ApplyXFMInputSpec