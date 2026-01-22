import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class MedianImage(MathsCommand):
    """Use fslmaths to generate a median image across a given dimension."""
    input_spec = MedianImageInput
    _suffix = '_median'