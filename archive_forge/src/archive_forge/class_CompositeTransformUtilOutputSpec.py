import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class CompositeTransformUtilOutputSpec(TraitedSpec):
    affine_transform = File(desc='Affine transform component')
    displacement_field = File(desc='Displacement field component')
    out_file = File(desc='Compound transformation file')