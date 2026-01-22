import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class MeanImage(MathsCommand):
    """Use fslmaths to generate a mean image across a given dimension."""
    input_spec = MeanImageInput
    _suffix = '_mean'