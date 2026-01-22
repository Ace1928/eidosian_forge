import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _regularization_constructor(self):
    return '--regularization {0}[{1},{2}]'.format(self.inputs.regularization, self.inputs.regularization_gradient_field_sigma, self.inputs.regularization_deformation_field_sigma)