import os
from ...base import (
class gtractResampleAnisotropyOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing the resampled transformed anisotropy image.', exists=True)