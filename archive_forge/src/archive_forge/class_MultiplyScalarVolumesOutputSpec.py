from nipype.interfaces.base import (
import os
class MultiplyScalarVolumesOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Volume1 * Volume2', exists=True)