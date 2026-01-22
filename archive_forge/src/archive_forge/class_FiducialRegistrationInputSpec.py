from nipype.interfaces.base import (
import os
class FiducialRegistrationInputSpec(CommandLineInputSpec):
    fixedLandmarks = InputMultiPath(traits.List(traits.Float(), minlen=3, maxlen=3), desc='Ordered list of landmarks in the fixed image', argstr='--fixedLandmarks %s...')
    movingLandmarks = InputMultiPath(traits.List(traits.Float(), minlen=3, maxlen=3), desc='Ordered list of landmarks in the moving image', argstr='--movingLandmarks %s...')
    saveTransform = traits.Either(traits.Bool, File(), hash_files=False, desc='Save the transform that results from registration', argstr='--saveTransform %s')
    transformType = traits.Enum('Translation', 'Rigid', 'Similarity', desc='Type of transform to produce', argstr='--transformType %s')
    rms = traits.Float(desc='Display RMS Error.', argstr='--rms %f')
    outputMessage = traits.Str(desc='Provides more information on the output', argstr='--outputMessage %s')