import os
from ...base import (
class BRAINSLandmarkInitializerInputSpec(CommandLineInputSpec):
    inputFixedLandmarkFilename = File(desc='input fixed landmark. *.fcsv', exists=True, argstr='--inputFixedLandmarkFilename %s')
    inputMovingLandmarkFilename = File(desc='input moving landmark. *.fcsv', exists=True, argstr='--inputMovingLandmarkFilename %s')
    inputWeightFilename = File(desc='Input weight file name for landmarks. Higher weighted landmark will be considered more heavily. Weights are proportional, that is the magnitude of weights will be normalized by its minimum and maximum value. ', exists=True, argstr='--inputWeightFilename %s')
    outputTransformFilename = traits.Either(traits.Bool, File(), hash_files=False, desc='output transform file name (ex: ./outputTransform.mat) ', argstr='--outputTransformFilename %s')