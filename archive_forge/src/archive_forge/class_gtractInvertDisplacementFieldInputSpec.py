import os
from ...base import (
class gtractInvertDisplacementFieldInputSpec(CommandLineInputSpec):
    baseImage = File(desc='Required: base image used to define the size of the inverse field', exists=True, argstr='--baseImage %s')
    deformationImage = File(desc='Required: Displacement field image', exists=True, argstr='--deformationImage %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: Output deformation field', argstr='--outputVolume %s')
    subsamplingFactor = traits.Int(desc='Subsampling factor for the deformation field', argstr='--subsamplingFactor %d')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')