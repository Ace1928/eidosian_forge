from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class AffSymTensor3DVol(CommandLineDtitk):
    """
    Applies affine transform to a tensor volume

    Example
    -------

    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.AffSymTensor3DVol()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.transform = 'im_affine.aff'
    >>> node.cmdline
    'affineSymTensor3DVolume -in im1.nii -interp LEI -out im1_affxfmd.nii
     -reorient PPD -trans im_affine.aff'
    >>> node.run() # doctest: +SKIP
    """
    input_spec = AffSymTensor3DVolInputSpec
    output_spec = AffSymTensor3DVolOutputSpec
    _cmd = 'affineSymTensor3DVolume'