import os
from ...base import (
class gtractAverageBvaluesInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Required: input image file name containing multiple baseline gradients to average', exists=True, argstr='--inputVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output NRRD file containing directly averaged baseline images', argstr='--outputVolume %s')
    directionsTolerance = traits.Float(desc='Tolerance for matching identical gradient direction pairs', argstr='--directionsTolerance %f')
    averageB0only = traits.Bool(desc='Average only baseline gradients. All other gradient directions are not averaged, but retained in the outputVolume', argstr='--averageB0only ')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')