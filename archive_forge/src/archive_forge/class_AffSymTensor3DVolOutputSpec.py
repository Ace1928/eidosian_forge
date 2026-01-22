from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class AffSymTensor3DVolOutputSpec(TraitedSpec):
    out_file = File(exists=True)