import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class TStat(AFNICommand):
    """Compute voxel-wise statistics using AFNI 3dTstat command

    For complete details, see the `3dTstat Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTstat.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> tstat = afni.TStat()
    >>> tstat.inputs.in_file = 'functional.nii'
    >>> tstat.inputs.args = '-mean'
    >>> tstat.inputs.out_file = 'stats'
    >>> tstat.cmdline
    '3dTstat -mean -prefix stats functional.nii'
    >>> res = tstat.run()  # doctest: +SKIP

    """
    _cmd = '3dTstat'
    input_spec = TStatInputSpec
    output_spec = AFNICommandOutputSpec