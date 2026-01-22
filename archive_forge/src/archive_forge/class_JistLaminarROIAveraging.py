import os
from ..base import (
class JistLaminarROIAveraging(SEMLikeCommandLine):
    """Compute an average profile over a given ROI."""
    input_spec = JistLaminarROIAveragingInputSpec
    output_spec = JistLaminarROIAveragingOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.laminar.JistLaminarROIAveraging '
    _outputs_filenames = {'outROI3': 'outROI3'}
    _redirect_x = True