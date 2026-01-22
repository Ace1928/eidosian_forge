import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class AR1ImageInput(MathsInput):
    dimension = traits.Enum('T', 'X', 'Y', 'Z', usedefault=True, argstr='-%sar1', position=4, desc='dimension to find AR(1) coefficientacross')