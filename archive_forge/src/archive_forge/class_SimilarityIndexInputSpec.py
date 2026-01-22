import os
from ...base import (
class SimilarityIndexInputSpec(CommandLineInputSpec):
    outputCSVFilename = File(desc='output CSV Filename', exists=True, argstr='--outputCSVFilename %s')
    ANNContinuousVolume = File(desc='ANN Continuous volume to be compared to the manual volume', exists=True, argstr='--ANNContinuousVolume %s')
    inputManualVolume = File(desc='input manual(reference) volume', exists=True, argstr='--inputManualVolume %s')
    thresholdInterval = traits.Float(desc='Threshold interval to compute similarity index between zero and one', argstr='--thresholdInterval %f')