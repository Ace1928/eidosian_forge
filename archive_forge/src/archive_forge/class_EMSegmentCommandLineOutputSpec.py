from nipype.interfaces.base import (
import os
class EMSegmentCommandLineOutputSpec(TraitedSpec):
    resultVolumeFileName = File(desc='The file name that the segmentation result volume will be written to.', exists=True)
    generateEmptyMRMLSceneAndQuit = File(desc='Used for testing.  Only write a scene with default mrml parameters.', exists=True)
    resultMRMLSceneFileName = File(desc='Write out the MRML scene after command line substitutions have been made.', exists=True)