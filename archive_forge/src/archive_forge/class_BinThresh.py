from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class BinThresh(CommandLineDtitk):
    """
    Binarizes an image.

    Example
    -------
    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.BinThresh()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.lower_bound = 0
    >>> node.inputs.upper_bound = 100
    >>> node.inputs.inside_value = 1
    >>> node.inputs.outside_value = 0
    >>> node.cmdline
    'BinaryThresholdImageFilter im1.nii im1_thrbin.nii 0 100 1 0'
    >>> node.run() # doctest: +SKIP

    """
    input_spec = BinThreshInputSpec
    output_spec = BinThreshOutputSpec
    _cmd = 'BinaryThresholdImageFilter'