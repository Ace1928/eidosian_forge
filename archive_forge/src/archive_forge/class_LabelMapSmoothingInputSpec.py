from nipype.interfaces.base import (
import os
class LabelMapSmoothingInputSpec(CommandLineInputSpec):
    labelToSmooth = traits.Int(desc='The label to smooth.  All others will be ignored.  If no label is selected by the user, the maximum label in the image is chosen by default.', argstr='--labelToSmooth %d')
    numberOfIterations = traits.Int(desc='The number of iterations of the level set AntiAliasing algorithm', argstr='--numberOfIterations %d')
    maxRMSError = traits.Float(desc='The maximum RMS error.', argstr='--maxRMSError %f')
    gaussianSigma = traits.Float(desc='The standard deviation of the Gaussian kernel', argstr='--gaussianSigma %f')
    inputVolume = File(position=-2, desc='Input label map to smooth', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Smoothed label map', argstr='%s')