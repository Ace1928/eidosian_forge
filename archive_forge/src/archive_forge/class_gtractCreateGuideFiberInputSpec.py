import os
from ...base import (
class gtractCreateGuideFiberInputSpec(CommandLineInputSpec):
    inputFiber = File(desc='Required: input fiber tract file name', exists=True, argstr='--inputFiber %s')
    numberOfPoints = traits.Int(desc='Number of points in output guide fiber', argstr='--numberOfPoints %d')
    outputFiber = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: output guide fiber file name', argstr='--outputFiber %s')
    writeXMLPolyDataFile = traits.Bool(desc='Flag to make use of XML files when reading and writing vtkPolyData.', argstr='--writeXMLPolyDataFile ')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')