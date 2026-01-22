from nipype.interfaces.base import (
import os
class GenerateAverageLmkFileInputSpec(CommandLineInputSpec):
    inputLandmarkFiles = InputMultiPath(traits.Str, desc='Input landmark files names (.fcsv or .wts)', sep=',', argstr='--inputLandmarkFiles %s')
    outputLandmarkFile = traits.Either(traits.Bool, File(), hash_files=False, desc='Output landmark file name that includes average values for landmarks (.fcsv or .wts)', argstr='--outputLandmarkFile %s')