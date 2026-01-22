import os
from ...utils.filemanip import split_filename
from ..base import (
class DTMetricOutputSpec(TraitedSpec):
    metric_stats = File(exists=True, desc='Diffusion Tensor statistics of the chosen metric')