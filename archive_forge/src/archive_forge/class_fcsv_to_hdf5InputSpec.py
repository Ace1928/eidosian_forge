import os
from ...base import (
class fcsv_to_hdf5InputSpec(CommandLineInputSpec):
    versionID = traits.Str(desc=',         Current version ID. It should be match with the version of BCD that will be using the output model file,       ', argstr='--versionID %s')
    landmarksInformationFile = traits.Either(traits.Bool, File(), hash_files=False, desc=',         name of HDF5 file to write matrices into,       ', argstr='--landmarksInformationFile %s')
    landmarkTypesList = File(desc=',         file containing list of landmark types,       ', exists=True, argstr='--landmarkTypesList %s')
    modelFile = traits.Either(traits.Bool, File(), hash_files=False, desc=',         name of HDF5 file containing BRAINSConstellationDetector Model file (LLSMatrices, LLSMeans and LLSSearchRadii),       ', argstr='--modelFile %s')
    landmarkGlobPattern = traits.Str(desc='Glob pattern to select fcsv files', argstr='--landmarkGlobPattern %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')