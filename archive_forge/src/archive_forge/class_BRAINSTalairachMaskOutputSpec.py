import os
from ...base import (
class BRAINSTalairachMaskOutputSpec(TraitedSpec):
    outputVolume = File(desc='Output filename for the resulting binary image', exists=True)