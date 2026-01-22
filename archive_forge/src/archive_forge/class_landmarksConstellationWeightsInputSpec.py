import os
from ...base import (
class landmarksConstellationWeightsInputSpec(CommandLineInputSpec):
    inputTrainingList = File(desc=',                 Setup file, giving all parameters for training up a Weight list for landmark.,             ', exists=True, argstr='--inputTrainingList %s')
    inputTemplateModel = File(desc='User-specified template model.,             ', exists=True, argstr='--inputTemplateModel %s')
    LLSModel = File(desc='Linear least squares model filename in HD5 format', exists=True, argstr='--LLSModel %s')
    outputWeightsList = traits.Either(traits.Bool, File(), hash_files=False, desc=',                 The filename of a csv file which is a list of landmarks and their corresponding weights.,             ', argstr='--outputWeightsList %s')