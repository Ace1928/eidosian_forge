import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _format_winsorize_image_intensities(self):
    if not self.inputs.winsorize_upper_quantile > self.inputs.winsorize_lower_quantile:
        raise RuntimeError('Upper bound MUST be more than lower bound: %g > %g' % (self.inputs.winsorize_upper_quantile, self.inputs.winsorize_lower_quantile))
    self._quantilesDone = True
    return '--winsorize-image-intensities [ %s, %s ]' % (self.inputs.winsorize_lower_quantile, self.inputs.winsorize_upper_quantile)