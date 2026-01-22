import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TNorm(AFNICommand):
    """Shifts voxel time series from input so that separate slices are aligned
    to the same temporal origin.

    For complete details, see the `3dTnorm Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTnorm.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> tnorm = afni.TNorm()
    >>> tnorm.inputs.in_file = 'functional.nii'
    >>> tnorm.inputs.norm2 = True
    >>> tnorm.inputs.out_file = 'rm.errts.unit errts+tlrc'
    >>> tnorm.cmdline
    '3dTnorm -norm2 -prefix rm.errts.unit errts+tlrc functional.nii'
    >>> res = tshift.run()  # doctest: +SKIP

    """
    _cmd = '3dTnorm'
    input_spec = TNormInputSpec
    output_spec = AFNICommandOutputSpec