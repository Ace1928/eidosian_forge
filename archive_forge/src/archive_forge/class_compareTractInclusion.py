import os
from ...base import (
class compareTractInclusion(SEMLikeCommandLine):
    """title: Compare Tracts

    category: Diffusion.GTRACT

    description: This program will halt with a status code indicating whether a test tract is nearly enough included in a standard tract in the sense that every fiber in the test tract has a low enough sum of squares distance to some fiber in the standard tract modulo spline resampling of every fiber to a fixed number of points.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = compareTractInclusionInputSpec
    output_spec = compareTractInclusionOutputSpec
    _cmd = ' compareTractInclusion '
    _outputs_filenames = {}
    _redirect_x = False