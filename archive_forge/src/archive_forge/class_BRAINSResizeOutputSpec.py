import os
from ...base import (
class BRAINSResizeOutputSpec(TraitedSpec):
    outputVolume = File(desc='Resulting scaled image', exists=True)