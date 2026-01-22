import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class MaskTool(AFNICommand):
    """3dmask_tool - for combining/dilating/eroding/filling masks

    For complete details, see the `3dmask_tool Documentation.
    <https://afni.nimh.nih.gov/pub../pub/dist/doc/program_help/3dmask_tool.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> masktool = afni.MaskTool()
    >>> masktool.inputs.in_file = 'functional.nii'
    >>> masktool.inputs.outputtype = 'NIFTI'
    >>> masktool.cmdline
    '3dmask_tool -prefix functional_mask.nii -input functional.nii'
    >>> res = automask.run()  # doctest: +SKIP

    """
    _cmd = '3dmask_tool'
    input_spec = MaskToolInputSpec
    output_spec = MaskToolOutputSpec