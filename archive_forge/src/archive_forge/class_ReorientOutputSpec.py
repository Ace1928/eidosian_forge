from ..utils.filemanip import fname_presuffix
from .base import SimpleInterface, TraitedSpec, BaseInterfaceInputSpec, traits, File
from .. import LooseVersion
class ReorientOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Reoriented image')
    transform = File(exists=True, desc='Affine transform from input orientation to output')