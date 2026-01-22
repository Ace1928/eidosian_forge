import os
from ...base import (
class GenerateSummedGradientImageInputSpec(CommandLineInputSpec):
    inputVolume1 = File(desc='input volume 1, usually t1 image', exists=True, argstr='--inputVolume1 %s')
    inputVolume2 = File(desc='input volume 2, usually t2 image', exists=True, argstr='--inputVolume2 %s')
    outputFileName = traits.Either(traits.Bool, File(), hash_files=False, desc='(required) output file name', argstr='--outputFileName %s')
    MaximumGradient = traits.Bool(desc='If set this flag, it will compute maximum gradient between two input volumes instead of sum of it.', argstr='--MaximumGradient ')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')