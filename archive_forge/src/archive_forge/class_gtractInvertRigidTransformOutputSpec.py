import os
from ...base import (
class gtractInvertRigidTransformOutputSpec(TraitedSpec):
    outputTransform = File(desc='Required: output transform file name', exists=True)