import os
from ..base import (
class RandomVol(SEMLikeCommandLine):
    """Generate a volume of random scalars."""
    input_spec = RandomVolInputSpec
    output_spec = RandomVolOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run edu.jhu.bme.smile.demo.RandomVol '
    _outputs_filenames = {'outRand1': 'outRand1.nii'}
    _redirect_x = True