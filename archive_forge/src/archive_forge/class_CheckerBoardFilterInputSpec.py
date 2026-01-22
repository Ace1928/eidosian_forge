from nipype.interfaces.base import (
import os
class CheckerBoardFilterInputSpec(CommandLineInputSpec):
    checkerPattern = InputMultiPath(traits.Int, desc='The pattern of input 1 and input 2 in the output image. The user can specify the number of checkers in each dimension. A checkerPattern of 2,2,1 means that images will alternate in every other checker in the first two dimensions. The same pattern will be used in the 3rd dimension.', sep=',', argstr='--checkerPattern %s')
    inputVolume1 = File(position=-3, desc='First Input volume', exists=True, argstr='%s')
    inputVolume2 = File(position=-2, desc='Second Input volume', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output filtered', argstr='%s')