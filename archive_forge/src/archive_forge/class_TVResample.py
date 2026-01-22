from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class TVResample(CommandLineDtitk):
    """
    Resamples a tensor volume.

    Example
    -------
    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.TVResample()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.target_file = 'im2.nii'
    >>> node.cmdline
    'TVResample -in im1.nii -out im1_resampled.nii -target im2.nii'
    >>> node.run() # doctest: +SKIP

    """
    input_spec = TVResampleInputSpec
    output_spec = TVResampleOutputSpec
    _cmd = 'TVResample'