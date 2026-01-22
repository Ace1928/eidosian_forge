import os
from ...base import (
class gtractTransformToDisplacementField(SEMLikeCommandLine):
    """title: Create Displacement Field

    category: Diffusion.GTRACT

    description: This program will compute forward deformation from the given Transform. The size of the DF is equal to MNI space

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta, Madhura Ingalhalikar, and Greg Harris

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractTransformToDisplacementFieldInputSpec
    output_spec = gtractTransformToDisplacementFieldOutputSpec
    _cmd = ' gtractTransformToDisplacementField '
    _outputs_filenames = {'outputDeformationFieldVolume': 'outputDeformationFieldVolume.nii'}
    _redirect_x = False