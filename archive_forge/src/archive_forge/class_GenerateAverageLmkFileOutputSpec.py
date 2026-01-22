from nipype.interfaces.base import (
import os
class GenerateAverageLmkFileOutputSpec(TraitedSpec):
    outputLandmarkFile = File(desc='Output landmark file name that includes average values for landmarks (.fcsv or .wts)', exists=True)