import os
from ...base import (
class gtractInvertRigidTransform(SEMLikeCommandLine):
    """title: Rigid Transform Inversion

    category: Diffusion.GTRACT

    description: This program will invert a Rigid transform.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractInvertRigidTransformInputSpec
    output_spec = gtractInvertRigidTransformOutputSpec
    _cmd = ' gtractInvertRigidTransform '
    _outputs_filenames = {'outputTransform': 'outputTransform.h5'}
    _redirect_x = False