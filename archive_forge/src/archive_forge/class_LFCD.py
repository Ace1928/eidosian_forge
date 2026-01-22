import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class LFCD(AFNICommand):
    """Performs degree centrality on a dataset using a given maskfile
    via the 3dLFCD command

    For complete details, see the `3dLFCD Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dLFCD.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> lfcd = afni.LFCD()
    >>> lfcd.inputs.in_file = 'functional.nii'
    >>> lfcd.inputs.mask = 'mask.nii'
    >>> lfcd.inputs.thresh = 0.8 # keep all connections with corr >= 0.8
    >>> lfcd.inputs.out_file = 'out.nii'
    >>> lfcd.cmdline
    '3dLFCD -mask mask.nii -prefix out.nii -thresh 0.800000 functional.nii'
    >>> res = lfcd.run()  # doctest: +SKIP
    """
    _cmd = '3dLFCD'
    input_spec = LFCDInputSpec
    output_spec = AFNICommandOutputSpec