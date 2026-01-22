import os
from ...base import (
class gtractCopyImageOrientationInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input file containing the signed short image to reorient without resampling.', exists=True, argstr='--inputVolume %s')
    inputReferenceVolume = File(desc='Required: input file containing orientation that will be cloned.', exists=True, argstr='--inputReferenceVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD or Nifti file containing the reoriented image in reference image space.', argstr='--outputVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')