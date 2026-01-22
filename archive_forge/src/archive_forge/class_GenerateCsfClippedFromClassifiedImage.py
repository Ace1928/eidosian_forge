import os
from ..base import (
class GenerateCsfClippedFromClassifiedImage(SEMLikeCommandLine):
    """title: GenerateCsfClippedFromClassifiedImage

    category: FeatureCreator

    description: Get the distance from a voxel to the nearest voxel of a given tissue type.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was written by Hans J. Johnson.
    """
    input_spec = GenerateCsfClippedFromClassifiedImageInputSpec
    output_spec = GenerateCsfClippedFromClassifiedImageOutputSpec
    _cmd = ' GenerateCsfClippedFromClassifiedImage '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False