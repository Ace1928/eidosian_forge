import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class StdImage(MathsCommand):
    """Use fslmaths to generate a standard deviation in an image across a given
    dimension.
    """
    input_spec = StdImageInput
    _suffix = '_std'