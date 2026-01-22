import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class AFNItoNIFTI(AFNICommand):
    """Converts AFNI format files to NIFTI format. This can also convert 2D or
    1D data, which you can numpy.squeeze() to remove extra dimensions.

    For complete details, see the `3dAFNItoNIFTI Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAFNItoNIFTI.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> a2n = afni.AFNItoNIFTI()
    >>> a2n.inputs.in_file = 'afni_output.3D'
    >>> a2n.inputs.out_file =  'afni_output.nii'
    >>> a2n.cmdline
    '3dAFNItoNIFTI -prefix afni_output.nii afni_output.3D'
    >>> res = a2n.run()  # doctest: +SKIP

    """
    _cmd = '3dAFNItoNIFTI'
    input_spec = AFNItoNIFTIInputSpec
    output_spec = AFNICommandOutputSpec

    def _overload_extension(self, value, name=None):
        path, base, ext = split_filename(value)
        if ext.lower() not in ['.nii', '.nii.gz', '.1d', '.1D']:
            ext += '.nii'
        return os.path.join(path, base + ext)

    def _gen_filename(self, name):
        return os.path.abspath(super(AFNItoNIFTI, self)._gen_filename(name))