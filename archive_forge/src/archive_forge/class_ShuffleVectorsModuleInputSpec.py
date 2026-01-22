import os
from ...base import (
class ShuffleVectorsModuleInputSpec(CommandLineInputSpec):
    inputVectorFileBaseName = File(desc='input vector file name prefix. Usually end with .txt and header file has prost fix of .txt.hdr', exists=True, argstr='--inputVectorFileBaseName %s')
    outputVectorFileBaseName = traits.Either(traits.Bool, File(), hash_files=False, desc='output vector file name prefix. Usually end with .txt and header file has prost fix of .txt.hdr', argstr='--outputVectorFileBaseName %s')
    resampleProportion = traits.Float(desc='downsample size of 1 will be the same size as the input images, downsample size of 3 will throw 2/3 the vectors away.', argstr='--resampleProportion %f')