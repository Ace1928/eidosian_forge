import os
from ...base import (
class CleanUpOverlapLabelsInputSpec(CommandLineInputSpec):
    inputBinaryVolumes = InputMultiPath(File(exists=True), desc='The list of binary images to be checked and cleaned up. Order is important. Binary volume given first always wins out. ', argstr='--inputBinaryVolumes %s...')
    outputBinaryVolumes = traits.Either(traits.Bool, InputMultiPath(File()), hash_files=False, desc='The output label map images, with integer values in it. Each label value specified in the inputLabels is combined into this output label map volume', argstr='--outputBinaryVolumes %s...')