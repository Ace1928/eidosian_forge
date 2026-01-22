import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EddyCorrect(FSLCommand):
    """

    .. warning:: Deprecated in FSL. Please use
      :class:`nipype.interfaces.fsl.epi.Eddy` instead

    Example
    -------

    >>> from nipype.interfaces.fsl import EddyCorrect
    >>> eddyc = EddyCorrect(in_file='diffusion.nii',
    ...                     out_file="diffusion_edc.nii", ref_num=0)
    >>> eddyc.cmdline
    'eddy_correct diffusion.nii diffusion_edc.nii 0'

    """
    _cmd = 'eddy_correct'
    input_spec = EddyCorrectInputSpec
    output_spec = EddyCorrectOutputSpec

    def __init__(self, **inputs):
        warnings.warn('Deprecated: Please use nipype.interfaces.fsl.epi.Eddy instead', DeprecationWarning)
        return super(EddyCorrect, self).__init__(**inputs)

    def _run_interface(self, runtime):
        runtime = super(EddyCorrect, self)._run_interface(runtime)
        if runtime.stderr:
            self.raise_exception(runtime)
        return runtime