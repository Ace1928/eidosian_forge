import os
from ..base import (
class JistBrainMp2rageDuraEstimation(SEMLikeCommandLine):
    """Filters a MP2RAGE brain image to obtain a probability map of dura matter."""
    input_spec = JistBrainMp2rageDuraEstimationInputSpec
    output_spec = JistBrainMp2rageDuraEstimationOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.brain.JistBrainMp2rageDuraEstimation '
    _outputs_filenames = {'outDura': 'outDura.nii'}
    _redirect_x = True