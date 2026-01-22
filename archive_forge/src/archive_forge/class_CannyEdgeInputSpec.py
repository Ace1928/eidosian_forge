import os
from ...base import (
class CannyEdgeInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input tissue label image', exists=True, argstr='--inputVolume %s')
    variance = traits.Float(desc='Variance and Maximum error are used in the Gaussian smoothing of the input image.  See  itkDiscreteGaussianImageFilter for information on these parameters.', argstr='--variance %f')
    upperThreshold = traits.Float(desc='Threshold is the lowest allowed value in the output image.  Its data type is the same as the data type of the output image. Any values below the Threshold level will be replaced with the OutsideValue parameter value, whose default is zero.  ', argstr='--upperThreshold %f')
    lowerThreshold = traits.Float(desc='Threshold is the lowest allowed value in the output image.  Its data type is the same as the data type of the output image. Any values below the Threshold level will be replaced with the OutsideValue parameter value, whose default is zero.  ', argstr='--lowerThreshold %f')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output image', argstr='--outputVolume %s')