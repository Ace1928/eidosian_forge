import os
from ...base import (
class BRAINSTransformFromFiducialsInputSpec(CommandLineInputSpec):
    fixedLandmarks = InputMultiPath(traits.List(traits.Float(), minlen=3, maxlen=3), desc='Ordered list of landmarks in the fixed image', argstr='--fixedLandmarks %s...')
    movingLandmarks = InputMultiPath(traits.List(traits.Float(), minlen=3, maxlen=3), desc='Ordered list of landmarks in the moving image', argstr='--movingLandmarks %s...')
    saveTransform = traits.Either(traits.Bool, File(), hash_files=False, desc='Save the transform that results from registration', argstr='--saveTransform %s')
    transformType = traits.Enum('Translation', 'Rigid', 'Similarity', desc='Type of transform to produce', argstr='--transformType %s')
    fixedLandmarksFile = File(desc='An fcsv formatted file with a list of landmark points.', exists=True, argstr='--fixedLandmarksFile %s')
    movingLandmarksFile = File(desc='An fcsv formatted file with a list of landmark points.', exists=True, argstr='--movingLandmarksFile %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')