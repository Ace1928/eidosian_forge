import os
from ...base import (
class gtractInvertBSplineTransformInputSpec(CommandLineInputSpec):
    inputReferenceVolume = File(desc='Required: input image file name to exemplify the anatomical space to interpolate over.', exists=True, argstr='--inputReferenceVolume %s')
    inputTransform = File(desc='Required: input B-Spline transform file name', exists=True, argstr='--inputTransform %s')
    outputTransform = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output transform file name', argstr='--outputTransform %s')
    landmarkDensity = InputMultiPath(traits.Int, desc='Number of landmark subdivisions in all 3 directions', sep=',', argstr='--landmarkDensity %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')