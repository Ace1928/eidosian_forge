import os
from ...base import (
class BRAINSTransformFromFiducialsOutputSpec(TraitedSpec):
    saveTransform = File(desc='Save the transform that results from registration', exists=True)