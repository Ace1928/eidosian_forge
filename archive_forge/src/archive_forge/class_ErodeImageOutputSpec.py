import os
from ...base import (
class ErodeImageOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)