import os
from ...base import (
class FlippedDifferenceOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)