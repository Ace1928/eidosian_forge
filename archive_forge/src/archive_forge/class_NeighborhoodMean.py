import os
from ...base import (
class NeighborhoodMean(SEMLikeCommandLine):
    """title: Neighborhood Mean

    category: Filtering.FeatureDetection

    description: Calculates the mean, for the given neighborhood size, at each voxel of the T1, T2, and FLAIR.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Mark Scully and Jeremy Bockholt.
    """
    input_spec = NeighborhoodMeanInputSpec
    output_spec = NeighborhoodMeanOutputSpec
    _cmd = ' NeighborhoodMean '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False