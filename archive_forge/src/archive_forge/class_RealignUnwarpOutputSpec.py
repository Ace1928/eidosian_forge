import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class RealignUnwarpOutputSpec(TraitedSpec):
    mean_image = File(exists=True, desc='Mean image file from the realignment & unwarping')
    modified_in_files = OutputMultiPath(traits.Either(traits.List(File(exists=True)), File(exists=True)), desc='Copies of all files passed to in_files. Headers will have been modified to align all images with the first, or optionally to first do that, extract a mean image, and re-align to that mean image.')
    realigned_unwarped_files = OutputMultiPath(traits.Either(traits.List(File(exists=True)), File(exists=True)), desc='Realigned and unwarped files written to disc.')
    realignment_parameters = OutputMultiPath(File(exists=True), desc='Estimated translation and rotation parameters')