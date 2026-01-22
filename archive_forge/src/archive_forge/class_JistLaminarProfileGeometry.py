import os
from ..base import (
class JistLaminarProfileGeometry(SEMLikeCommandLine):
    """Compute various geometric quantities for a cortical layers."""
    input_spec = JistLaminarProfileGeometryInputSpec
    output_spec = JistLaminarProfileGeometryOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.laminar.JistLaminarProfileGeometry '
    _outputs_filenames = {'outResult': 'outResult.nii'}
    _redirect_x = True