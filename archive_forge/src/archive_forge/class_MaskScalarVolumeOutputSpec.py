from nipype.interfaces.base import (
import os
class MaskScalarVolumeOutputSpec(TraitedSpec):
    OutputVolume = File(position=-1, desc='Output volume: Input Volume masked by label value from Mask Volume', exists=True)