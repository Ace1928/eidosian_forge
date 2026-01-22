import os
from ...base import (
class CannyEdgeOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: output image', exists=True)