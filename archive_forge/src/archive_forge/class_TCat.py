import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class TCat(AFNICommand):
    """Concatenate sub-bricks from input datasets into one big 3D+time dataset.

    TODO Replace InputMultiPath in_files with Traits.List, if possible. Current
    version adds extra whitespace.

    For complete details, see the `3dTcat Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTcat.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> tcat = afni.TCat()
    >>> tcat.inputs.in_files = ['functional.nii', 'functional2.nii']
    >>> tcat.inputs.out_file= 'functional_tcat.nii'
    >>> tcat.inputs.rlt = '+'
    >>> tcat.cmdline
    '3dTcat -rlt+ -prefix functional_tcat.nii functional.nii functional2.nii'
    >>> res = tcat.run()  # doctest: +SKIP

    """
    _cmd = '3dTcat'
    input_spec = TCatInputSpec
    output_spec = AFNICommandOutputSpec