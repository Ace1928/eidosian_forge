import os
from ...base import (
class gtractTransformToDisplacementFieldOutputSpec(TraitedSpec):
    outputDeformationFieldVolume = File(desc='Output deformation field', exists=True)