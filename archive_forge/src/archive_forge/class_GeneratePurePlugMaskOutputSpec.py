import os
from ...base import (
class GeneratePurePlugMaskOutputSpec(TraitedSpec):
    outputMaskFile = File(desc='(required) Output binary mask file name', exists=True)