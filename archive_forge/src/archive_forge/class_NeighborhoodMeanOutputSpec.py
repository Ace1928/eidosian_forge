import os
from ...base import (
class NeighborhoodMeanOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)