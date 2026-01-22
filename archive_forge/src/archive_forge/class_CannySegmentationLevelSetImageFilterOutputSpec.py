import os
from ...base import (
class CannySegmentationLevelSetImageFilterOutputSpec(TraitedSpec):
    outputVolume = File(exists=True)
    outputSpeedVolume = File(exists=True)