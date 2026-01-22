import os
from ..base import TraitedSpec, File, traits
from .base import ANTSCommand, ANTSCommandInputSpec
class ConvertScalarImageToRGBOutputSpec(TraitedSpec):
    output_image = File(exists=True, desc='converted RGB image')