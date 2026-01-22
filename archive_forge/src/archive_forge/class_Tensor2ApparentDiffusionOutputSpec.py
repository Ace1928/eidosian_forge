import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class Tensor2ApparentDiffusionOutputSpec(TraitedSpec):
    ADC = File(exists=True, desc='the output image of the major eigenvectors of the diffusion tensor image.')