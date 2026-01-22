import os
from ...base import (
class DilateMaskOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)