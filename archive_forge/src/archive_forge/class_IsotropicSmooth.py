import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class IsotropicSmooth(MathsCommand):
    """Use fslmaths to spatially smooth an image with a gaussian kernel."""
    input_spec = IsotropicSmoothInput
    _suffix = '_smooth'

    def _format_arg(self, name, spec, value):
        if name == 'fwhm':
            sigma = float(value) / np.sqrt(8 * np.log(2))
            return spec.argstr % sigma
        return super(IsotropicSmooth, self)._format_arg(name, spec, value)