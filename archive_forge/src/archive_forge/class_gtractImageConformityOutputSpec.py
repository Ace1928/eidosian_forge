import os
from ...base import (
class gtractImageConformityOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output Nrrd or Nifti file containing the reoriented image in reference image space.', exists=True)