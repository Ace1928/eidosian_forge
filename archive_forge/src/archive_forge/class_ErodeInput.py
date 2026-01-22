import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class ErodeInput(KernelInput):
    minimum_filter = traits.Bool(argstr='%s', position=6, usedefault=True, default_value=False, desc='if true, minimum filter rather than erosion by zeroing-out')