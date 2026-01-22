import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class DARTELInputSpec(SPMCommandInputSpec):
    image_files = traits.List(traits.List(ImageFileSPM(exists=True)), desc='A list of files to be segmented', field='warp.images', copyfile=False, mandatory=True)
    template_prefix = traits.Str('Template', usedefault=True, field='warp.settings.template', desc='Prefix for template')
    regularization_form = traits.Enum('Linear', 'Membrane', 'Bending', field='warp.settings.rform', desc='Form of regularization energy term')
    iteration_parameters = traits.List(traits.Tuple(traits.Range(1, 10), traits.Tuple(traits.Float, traits.Float, traits.Float), traits.Enum(1, 2, 4, 8, 16, 32, 64, 128, 256, 512), traits.Enum(0, 0.5, 1, 2, 4, 8, 16, 32)), minlen=3, maxlen=12, field='warp.settings.param', desc='List of tuples for each iteration\n\n  * Inner iterations\n  * Regularization parameters\n  * Time points for deformation model\n  * smoothing parameter\n\n')
    optimization_parameters = traits.Tuple(traits.Float, traits.Range(1, 8), traits.Range(1, 8), field='warp.settings.optim', desc='Optimization settings a tuple:\n\n  * LM regularization\n  * cycles of multigrid solver\n  * relaxation iterations\n\n')