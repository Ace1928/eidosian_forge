import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
@staticmethod
def _remove_extension(in_file):
    dn, bn, _ = split_filename(in_file)
    return os.path.join(dn, bn)