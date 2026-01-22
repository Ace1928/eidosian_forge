from nipype.interfaces.base import (
import os
class EMSegmentTransformToNewFormatOutputSpec(TraitedSpec):
    outputMRMLFileName = File(desc='Write out the MRML scene after transformation to format 3.6.3 has been made. - has to be in the same directory as the input MRML file due to Slicer Core bug  - please include absolute  file name in path ', exists=True)