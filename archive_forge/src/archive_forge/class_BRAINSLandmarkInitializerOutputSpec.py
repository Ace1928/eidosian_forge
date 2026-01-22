import os
from ...base import (
class BRAINSLandmarkInitializerOutputSpec(TraitedSpec):
    outputTransformFilename = File(desc='output transform file name (ex: ./outputTransform.mat) ', exists=True)