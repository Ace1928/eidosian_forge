import os
from ..base import TraitedSpec, File, traits
from .base import ANTSCommand, ANTSCommandInputSpec
class CreateTiledMosaicOutputSpec(TraitedSpec):
    output_image = File(exists=True, desc='image file')