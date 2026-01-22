import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TransformFSLConvertOutputSpec(TraitedSpec):
    out_transform = File(exists=True, desc="output transformed affine in mrtrix3's format")