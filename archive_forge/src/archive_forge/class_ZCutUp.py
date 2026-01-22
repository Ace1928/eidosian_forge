import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ZCutUp(AFNICommand):
    """Cut z-slices from a volume using AFNI 3dZcutup command

    For complete details, see the `3dZcutup Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dZcutup.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> zcutup = afni.ZCutUp()
    >>> zcutup.inputs.in_file = 'functional.nii'
    >>> zcutup.inputs.out_file = 'functional_zcutup.nii'
    >>> zcutup.inputs.keep= '0 10'
    >>> zcutup.cmdline
    '3dZcutup -keep 0 10 -prefix functional_zcutup.nii functional.nii'
    >>> res = zcutup.run()  # doctest: +SKIP

    """
    _cmd = '3dZcutup'
    input_spec = ZCutUpInputSpec
    output_spec = AFNICommandOutputSpec