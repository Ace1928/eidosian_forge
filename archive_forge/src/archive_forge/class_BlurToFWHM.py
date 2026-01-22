import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class BlurToFWHM(AFNICommand):
    """Blurs a 'master' dataset until it reaches a specified FWHM smoothness
    (approximately).

    For complete details, see the `3dBlurToFWHM Documentation
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dBlurToFWHM.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> blur = afni.preprocess.BlurToFWHM()
    >>> blur.inputs.in_file = 'epi.nii'
    >>> blur.inputs.fwhm = 2.5
    >>> blur.cmdline  # doctest: +ELLIPSIS
    '3dBlurToFWHM -FWHM 2.500000 -input epi.nii -prefix epi_afni'
    >>> res = blur.run()  # doctest: +SKIP

    """
    _cmd = '3dBlurToFWHM'
    input_spec = BlurToFWHMInputSpec
    output_spec = AFNICommandOutputSpec