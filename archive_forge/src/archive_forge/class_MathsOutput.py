import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class MathsOutput(TraitedSpec):
    """Output Spec for seg_maths interfaces."""
    out_file = File(desc='image written after calculations')