import os
from ...base import (
class DumpBinaryTrainingVectors(SEMLikeCommandLine):
    """title: Erode Image

    category: Filtering.FeatureDetection

    description: Uses mathematical morphology to erode the input images.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Mark Scully and Jeremy Bockholt.
    """
    input_spec = DumpBinaryTrainingVectorsInputSpec
    output_spec = DumpBinaryTrainingVectorsOutputSpec
    _cmd = ' DumpBinaryTrainingVectors '
    _outputs_filenames = {}
    _redirect_x = False