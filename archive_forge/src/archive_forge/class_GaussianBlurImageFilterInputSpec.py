from nipype.interfaces.base import (
import os
class GaussianBlurImageFilterInputSpec(CommandLineInputSpec):
    sigma = traits.Float(desc='Sigma value in physical units (e.g., mm) of the Gaussian kernel', argstr='--sigma %f')
    inputVolume = File(position=-2, desc='Input volume', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Blurred Volume', argstr='%s')