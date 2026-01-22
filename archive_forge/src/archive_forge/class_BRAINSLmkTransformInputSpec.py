import os
from ...base import (
class BRAINSLmkTransformInputSpec(CommandLineInputSpec):
    inputMovingLandmarks = File(desc='Input Moving Landmark list file in fcsv,             ', exists=True, argstr='--inputMovingLandmarks %s')
    inputFixedLandmarks = File(desc='Input Fixed Landmark list file in fcsv,             ', exists=True, argstr='--inputFixedLandmarks %s')
    outputAffineTransform = traits.Either(traits.Bool, File(), hash_files=False, desc='The filename for the estimated affine transform,             ', argstr='--outputAffineTransform %s')
    inputMovingVolume = File(desc='The filename of input moving volume', exists=True, argstr='--inputMovingVolume %s')
    inputReferenceVolume = File(desc='The filename of the reference volume', exists=True, argstr='--inputReferenceVolume %s')
    outputResampledVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='The filename of the output resampled volume', argstr='--outputResampledVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')