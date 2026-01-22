import os
from ...base import (
class BRAINSTrimForegroundInDirectionOutputSpec(TraitedSpec):
    outputVolume = File(desc='Output image with neck and air-filling noise trimmed isotropic image with AC at center of image.', exists=True)