import os
from ...base import (
class gtractClipAnisotropyOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing the clipped anisotropy image', exists=True)