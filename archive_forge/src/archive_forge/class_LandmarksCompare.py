from nipype.interfaces.base import (
import os
class LandmarksCompare(SEMLikeCommandLine):
    """title: Compare Fiducials

    category: Testing

    description: Compares two .fcsv or .wts text files and verifies that they are identicle.  Used for testing landmarks files.

    contributor: Ali Ghayoor
    """
    input_spec = LandmarksCompareInputSpec
    output_spec = LandmarksCompareOutputSpec
    _cmd = ' LandmarksCompare '
    _outputs_filenames = {}
    _redirect_x = False