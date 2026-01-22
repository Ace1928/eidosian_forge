import os
from ..base import (
class JistLaminarProfileSampling(SEMLikeCommandLine):
    """Sample some intensity image along a cortical profile across layer surfaces."""
    input_spec = JistLaminarProfileSamplingInputSpec
    output_spec = JistLaminarProfileSamplingOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.laminar.JistLaminarProfileSampling '
    _outputs_filenames = {'outProfile2': 'outProfile2.nii', 'outProfilemapped': 'outProfilemapped.nii'}
    _redirect_x = True