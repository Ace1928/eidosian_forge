from nipype.interfaces.base import (
import os
class MultiResolutionAffineRegistrationInputSpec(CommandLineInputSpec):
    fixedImage = File(position=-2, desc='Image which defines the space into which the moving image is registered', exists=True, argstr='%s')
    movingImage = File(position=-1, desc="The transform goes from the fixed image's space into the moving image's space", exists=True, argstr='%s')
    resampledImage = traits.Either(traits.Bool, File(), hash_files=False, desc='Registration results', argstr='--resampledImage %s')
    saveTransform = traits.Either(traits.Bool, File(), hash_files=False, desc='Save the output transform from the registration', argstr='--saveTransform %s')
    fixedImageMask = File(desc='Label image which defines a mask of interest for the fixed image', exists=True, argstr='--fixedImageMask %s')
    fixedImageROI = traits.List(desc='Label image which defines a ROI of interest for the fixed image', argstr='--fixedImageROI %s')
    numIterations = traits.Int(desc='Number of iterations to run at each resolution level.', argstr='--numIterations %d')
    numLineIterations = traits.Int(desc='Number of iterations to run at each resolution level.', argstr='--numLineIterations %d')
    stepSize = traits.Float(desc='The maximum step size of the optimizer in voxels', argstr='--stepSize %f')
    stepTolerance = traits.Float(desc='The maximum step size of the optimizer in voxels', argstr='--stepTolerance %f')
    metricTolerance = traits.Float(argstr='--metricTolerance %f')