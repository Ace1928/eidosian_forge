import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegResampleOutputSpec(TraitedSpec):
    """Output Spec for RegResample."""
    out_file = File(desc='The output filename of the transformed image')