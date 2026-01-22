import os
from ...base import (
class gtractResampleB0InputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input file containing the 4D image', exists=True, argstr='--inputVolume %s')
    inputAnatomicalVolume = File(desc='Required: input file containing the anatomical image defining the origin, spacing and size of the resampled image (template)', exists=True, argstr='--inputAnatomicalVolume %s')
    inputTransform = File(desc='Required: input Rigid OR Bspline transform file name', exists=True, argstr='--inputTransform %s')
    vectorIndex = traits.Int(desc='Index in the diffusion weighted image set for the B0 image', argstr='--vectorIndex %d')
    transformType = traits.Enum('Rigid', 'B-Spline', desc='Transform type: Rigid, B-Spline', argstr='--transformType %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing the resampled input image.', argstr='--outputVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')