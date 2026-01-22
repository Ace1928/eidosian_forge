import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class SegmentOutputSpec(TraitedSpec):
    native_gm_image = File(desc='native space grey probability map')
    normalized_gm_image = File(desc='normalized grey probability map')
    modulated_gm_image = File(desc='modulated, normalized grey probability map')
    native_wm_image = File(desc='native space white probability map')
    normalized_wm_image = File(desc='normalized white probability map')
    modulated_wm_image = File(desc='modulated, normalized white probability map')
    native_csf_image = File(desc='native space csf probability map')
    normalized_csf_image = File(desc='normalized csf probability map')
    modulated_csf_image = File(desc='modulated, normalized csf probability map')
    modulated_input_image = File(deprecated='0.10', new_name='bias_corrected_image', desc='bias-corrected version of input image')
    bias_corrected_image = File(desc='bias-corrected version of input image')
    transformation_mat = File(exists=True, desc='Normalization transformation')
    inverse_transformation_mat = File(exists=True, desc='Inverse normalization info')