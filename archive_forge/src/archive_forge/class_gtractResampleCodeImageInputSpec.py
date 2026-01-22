import os
from ...base import (
class gtractResampleCodeImageInputSpec(CommandLineInputSpec):
    inputCodeVolume = File(desc='Required: input file containing the code image', exists=True, argstr='--inputCodeVolume %s')
    inputReferenceVolume = File(desc='Required: input file containing the standard image to clone the characteristics of.', exists=True, argstr='--inputReferenceVolume %s')
    inputTransform = File(desc='Required: input Rigid or Inverse-B-Spline transform file name', exists=True, argstr='--inputTransform %s')
    transformType = traits.Enum('Rigid', 'Affine', 'B-Spline', 'Inverse-B-Spline', 'None', desc='Transform type: Rigid or Inverse-B-Spline', argstr='--transformType %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing the resampled code image in acquisition space.', argstr='--outputVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')