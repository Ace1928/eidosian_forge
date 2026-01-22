import os
from ...base import (
class gtractInvertDisplacementFieldOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: Output deformation field', exists=True)