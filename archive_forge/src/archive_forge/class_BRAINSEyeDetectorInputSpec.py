import os
from ...base import (
class BRAINSEyeDetectorInputSpec(CommandLineInputSpec):
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')
    inputVolume = File(desc='The input volume', exists=True, argstr='--inputVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='The output volume', argstr='--outputVolume %s')
    debugDir = traits.Str(desc='A place for debug information', argstr='--debugDir %s')