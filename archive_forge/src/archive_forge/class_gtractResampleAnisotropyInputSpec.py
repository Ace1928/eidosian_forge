import os
from ...base import (
class gtractResampleAnisotropyInputSpec(CommandLineInputSpec):
    inputAnisotropyVolume = File(desc='Required: input file containing the anisotropy image', exists=True, argstr='--inputAnisotropyVolume %s')
    inputAnatomicalVolume = File(desc='Required: input file containing the anatomical image whose characteristics will be cloned.', exists=True, argstr='--inputAnatomicalVolume %s')
    inputTransform = File(desc='Required: input Rigid OR Bspline transform file name', exists=True, argstr='--inputTransform %s')
    transformType = traits.Enum('Rigid', 'B-Spline', desc='Transform type: Rigid, B-Spline', argstr='--transformType %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing the resampled transformed anisotropy image.', argstr='--outputVolume %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')