import os
from ...base import (
class gtractResampleB0(SEMLikeCommandLine):
    """title: Resample B0

    category: Diffusion.GTRACT

    description: This program will resample a signed short image using either a Rigid or B-Spline transform. The user must specify a template image that will be used to define the origin, orientation, spacing, and size of the resampled image.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractResampleB0InputSpec
    output_spec = gtractResampleB0OutputSpec
    _cmd = ' gtractResampleB0 '
    _outputs_filenames = {'outputVolume': 'outputVolume.nrrd'}
    _redirect_x = False