from nipype.interfaces.base import (
import os
class SubtractScalarVolumesOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Volume1 - Volume2', exists=True)