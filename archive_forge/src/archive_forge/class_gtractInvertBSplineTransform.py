import os
from ...base import (
class gtractInvertBSplineTransform(SEMLikeCommandLine):
    """title: B-Spline Transform Inversion

    category: Diffusion.GTRACT

    description: This program will invert a B-Spline transform using a thin-plate spline approximation.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractInvertBSplineTransformInputSpec
    output_spec = gtractInvertBSplineTransformOutputSpec
    _cmd = ' gtractInvertBSplineTransform '
    _outputs_filenames = {'outputTransform': 'outputTransform.h5'}
    _redirect_x = False