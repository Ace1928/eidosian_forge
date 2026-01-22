import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class PercentileImageInput(MathsInput):
    dimension = traits.Enum('T', 'X', 'Y', 'Z', usedefault=True, argstr='-%sperc', position=4, desc='dimension to percentile across')
    perc = traits.Range(low=0, high=100, argstr='%f', position=5, desc='nth percentile (0-100) of FULL RANGE across dimension')