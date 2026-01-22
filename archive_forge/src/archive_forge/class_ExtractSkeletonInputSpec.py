from nipype.interfaces.base import (
import os
class ExtractSkeletonInputSpec(CommandLineInputSpec):
    InputImageFileName = File(position=-2, desc='Input image', exists=True, argstr='%s')
    OutputImageFileName = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Skeleton of the input image', argstr='%s')
    type = traits.Enum('1D', '2D', desc='Type of skeleton to create', argstr='--type %s')
    dontPrune = traits.Bool(desc='Return the full skeleton, not just the maximal skeleton', argstr='--dontPrune ')
    numPoints = traits.Int(desc='Number of points used to represent the skeleton', argstr='--numPoints %d')
    pointsFile = traits.Str(desc='Name of the file to store the coordinates of the central (1D) skeleton points', argstr='--pointsFile %s')