from nipype.interfaces.base import (
import os
class OrientScalarVolumeOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='The oriented volume', exists=True)