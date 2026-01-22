import os
from ..base import (
class RandomVolInputSpec(CommandLineInputSpec):
    inSize = traits.Int(desc='Size of Volume in X direction', argstr='--inSize %d')
    inSize2 = traits.Int(desc='Size of Volume in Y direction', argstr='--inSize2 %d')
    inSize3 = traits.Int(desc='Size of Volume in Z direction', argstr='--inSize3 %d')
    inSize4 = traits.Int(desc='Size of Volume in t direction', argstr='--inSize4 %d')
    inStandard = traits.Int(desc='Standard Deviation for Normal Distribution', argstr='--inStandard %d')
    inLambda = traits.Float(desc='Lambda Value for Exponential Distribution', argstr='--inLambda %f')
    inMaximum = traits.Int(desc='Maximum Value', argstr='--inMaximum %d')
    inMinimum = traits.Int(desc='Minimum Value', argstr='--inMinimum %d')
    inField = traits.Enum('Uniform', 'Normal', 'Exponential', desc='Field', argstr='--inField %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outRand1 = traits.Either(traits.Bool, File(), hash_files=False, desc='Rand1', argstr='--outRand1 %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)