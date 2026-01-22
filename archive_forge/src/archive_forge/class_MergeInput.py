import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class MergeInput(MathsInput):
    """Input Spec for seg_maths merge operation."""
    dimension = traits.Int(mandatory=True, desc='Dimension to merge the images.')
    merge_files = traits.List(File(exists=True), argstr='%s', mandatory=True, position=4, desc='List of images to merge to the working image <input>.')