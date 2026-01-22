import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class ApplyDeformationFieldInputSpec(SPMCommandInputSpec):
    in_files = InputMultiPath(ImageFileSPM(exists=True), mandatory=True, field='fnames')
    deformation_field = File(exists=True, mandatory=True, field='comp{1}.def')
    reference_volume = ImageFileSPM(exists=True, mandatory=True, field='comp{2}.id.space')
    interp = traits.Range(low=0, high=7, field='interp', desc='degree of b-spline used for interpolation')