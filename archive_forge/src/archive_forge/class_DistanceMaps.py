import os
from ...base import (
class DistanceMaps(SEMLikeCommandLine):
    """title: Mauerer Distance

    category: Filtering.FeatureDetection

    description: Get the distance from a voxel to the nearest voxel of a given tissue type.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Mark Scully and Jeremy Bockholt.
    """
    input_spec = DistanceMapsInputSpec
    output_spec = DistanceMapsOutputSpec
    _cmd = ' DistanceMaps '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False