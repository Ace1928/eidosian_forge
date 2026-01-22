import os
from ...base import (
class extractNrrdVectorIndex(SEMLikeCommandLine):
    """title: Extract Nrrd Index

    category: Diffusion.GTRACT

    description: This program will extract a 3D image (single vector) from a vector 3D image at a given vector index.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = extractNrrdVectorIndexInputSpec
    output_spec = extractNrrdVectorIndexOutputSpec
    _cmd = ' extractNrrdVectorIndex '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False