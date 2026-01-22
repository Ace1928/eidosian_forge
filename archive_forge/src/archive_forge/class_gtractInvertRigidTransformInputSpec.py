import os
from ...base import (
class gtractInvertRigidTransformInputSpec(CommandLineInputSpec):
    inputTransform = File(desc='Required: input rigid transform file name', exists=True, argstr='--inputTransform %s')
    outputTransform = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output transform file name', argstr='--outputTransform %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')