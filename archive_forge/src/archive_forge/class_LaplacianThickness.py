import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class LaplacianThickness(ANTSCommand):
    """Calculates the cortical thickness from an anatomical image

    Examples
    --------

    >>> from nipype.interfaces.ants import LaplacianThickness
    >>> cort_thick = LaplacianThickness()
    >>> cort_thick.inputs.input_wm = 'white_matter.nii.gz'
    >>> cort_thick.inputs.input_gm = 'gray_matter.nii.gz'
    >>> cort_thick.cmdline
    'LaplacianThickness white_matter.nii.gz gray_matter.nii.gz white_matter_thickness.nii.gz'

    >>> cort_thick.inputs.output_image = 'output_thickness.nii.gz'
    >>> cort_thick.cmdline
    'LaplacianThickness white_matter.nii.gz gray_matter.nii.gz output_thickness.nii.gz'

    """
    _cmd = 'LaplacianThickness'
    input_spec = LaplacianThicknessInputSpec
    output_spec = LaplacianThicknessOutputSpec