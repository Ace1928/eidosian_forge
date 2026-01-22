import os
from ...base import (
class gtractCreateGuideFiber(SEMLikeCommandLine):
    """title: Create Guide Fiber

    category: Diffusion.GTRACT

    description: This program will create a guide fiber by averaging fibers from a previously generated tract.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractCreateGuideFiberInputSpec
    output_spec = gtractCreateGuideFiberOutputSpec
    _cmd = ' gtractCreateGuideFiber '
    _outputs_filenames = {'outputFiber': 'outputFiber.vtk'}
    _redirect_x = False