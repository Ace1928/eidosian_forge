import os
from ...base import (
class DilateImageOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)