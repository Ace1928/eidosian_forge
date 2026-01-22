from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class SVResample(CommandLineDtitk):
    """
    Resamples a scalar volume.

    Example
    -------
    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.SVResample()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.target_file = 'im2.nii'
    >>> node.cmdline
    'SVResample -in im1.nii -out im1_resampled.nii -target im2.nii'
    >>> node.run() # doctest: +SKIP

    """
    input_spec = SVResampleInputSpec
    output_spec = SVResampleOutputSpec
    _cmd = 'SVResample'