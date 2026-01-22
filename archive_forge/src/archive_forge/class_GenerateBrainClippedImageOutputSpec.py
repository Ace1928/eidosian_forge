import os
from ...base import (
class GenerateBrainClippedImageOutputSpec(TraitedSpec):
    outputFileName = File(desc='(required) output file name', exists=True)