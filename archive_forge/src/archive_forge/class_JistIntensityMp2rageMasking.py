import os
from ..base import (
class JistIntensityMp2rageMasking(SEMLikeCommandLine):
    """Estimate a background signal mask for a MP2RAGE dataset."""
    input_spec = JistIntensityMp2rageMaskingInputSpec
    output_spec = JistIntensityMp2rageMaskingOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.intensity.JistIntensityMp2rageMasking '
    _outputs_filenames = {'outSignal2': 'outSignal2.nii', 'outSignal': 'outSignal.nii', 'outMasked2': 'outMasked2.nii', 'outMasked': 'outMasked.nii'}
    _redirect_x = True