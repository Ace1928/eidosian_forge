import os
from ...base import (
class NeighborhoodMedianOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)