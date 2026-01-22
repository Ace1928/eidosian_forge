from nipype.interfaces.base import (
import os
class ResampleScalarVectorDWIVolumeOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Resampled Volume', exists=True)