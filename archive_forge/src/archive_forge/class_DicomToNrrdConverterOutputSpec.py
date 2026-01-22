from nipype.interfaces.base import (
import os
class DicomToNrrdConverterOutputSpec(TraitedSpec):
    outputDirectory = Directory(desc='Directory holding the output NRRD format', exists=True)