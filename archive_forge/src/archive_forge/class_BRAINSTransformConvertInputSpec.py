import os
from ...base import (
class BRAINSTransformConvertInputSpec(CommandLineInputSpec):
    inputTransform = File(exists=True, argstr='--inputTransform %s')
    referenceVolume = File(exists=True, argstr='--referenceVolume %s')
    outputTransformType = traits.Enum('Affine', 'VersorRigid', 'ScaleVersor', 'ScaleSkewVersor', 'DisplacementField', 'Same', desc='The target transformation type. Must be conversion-compatible with the input transform type', argstr='--outputTransformType %s')
    outputPrecisionType = traits.Enum('double', 'float', desc='Precision type of the output transform. It can be either single precision or double precision', argstr='--outputPrecisionType %s')
    displacementVolume = traits.Either(traits.Bool, File(), hash_files=False, argstr='--displacementVolume %s')
    outputTransform = traits.Either(traits.Bool, File(), hash_files=False, argstr='--outputTransform %s')