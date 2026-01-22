import os
from ...base import (
class TextureMeasureFilterOutputSpec(TraitedSpec):
    outputFilename = File(exists=True)