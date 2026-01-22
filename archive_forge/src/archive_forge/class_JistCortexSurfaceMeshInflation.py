import os
from ..base import (
class JistCortexSurfaceMeshInflation(SEMLikeCommandLine):
    """Inflates a cortical surface mesh.

    References
    ----------
    D. Tosun, M. E. Rettmann, X. Han, X. Tao, C. Xu, S. M. Resnick, D. Pham, and J. L. Prince,
    Cortical Surface Segmentation and Mapping, NeuroImage, vol. 23, pp. S108--S118, 2004.

    """
    input_spec = JistCortexSurfaceMeshInflationInputSpec
    output_spec = JistCortexSurfaceMeshInflationOutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run de.mpg.cbs.jist.cortex.JistCortexSurfaceMeshInflation '
    _outputs_filenames = {'outOriginal': 'outOriginal', 'outInflated': 'outInflated'}
    _redirect_x = True