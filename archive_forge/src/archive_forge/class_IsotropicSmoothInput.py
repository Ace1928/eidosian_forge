import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class IsotropicSmoothInput(MathsInput):
    fwhm = traits.Float(mandatory=True, xor=['sigma'], position=4, argstr='-s %.5f', desc='fwhm of smoothing kernel [mm]')
    sigma = traits.Float(mandatory=True, xor=['fwhm'], position=4, argstr='-s %.5f', desc='sigma of smoothing kernel [mm]')