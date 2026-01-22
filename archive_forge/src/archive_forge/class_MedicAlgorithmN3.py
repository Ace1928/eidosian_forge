import os
from ..base import (
class MedicAlgorithmN3(SEMLikeCommandLine):
    """Non-parametric Intensity Non-uniformity Correction, N3, originally by J.G. Sled."""
    input_spec = MedicAlgorithmN3InputSpec
    output_spec = MedicAlgorithmN3OutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run edu.jhu.ece.iacl.plugins.classification.MedicAlgorithmN3 '
    _outputs_filenames = {'outInhomogeneity2': 'outInhomogeneity2.nii', 'outInhomogeneity': 'outInhomogeneity.nii'}
    _redirect_x = True