import os
from ...base import (
class DumpBinaryTrainingVectorsInputSpec(CommandLineInputSpec):
    inputHeaderFilename = File(desc='Required: input header file name', exists=True, argstr='--inputHeaderFilename %s')
    inputVectorFilename = File(desc='Required: input vector filename', exists=True, argstr='--inputVectorFilename %s')