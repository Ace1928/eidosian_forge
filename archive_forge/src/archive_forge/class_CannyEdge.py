import os
from ...base import (
class CannyEdge(SEMLikeCommandLine):
    """title: Canny Edge Detection

    category: Filtering.FeatureDetection

    description: Get the distance from a voxel to the nearest voxel of a given tissue type.

    version: 0.1.0.(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was written by Hans J. Johnson.
    """
    input_spec = CannyEdgeInputSpec
    output_spec = CannyEdgeOutputSpec
    _cmd = ' CannyEdge '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False