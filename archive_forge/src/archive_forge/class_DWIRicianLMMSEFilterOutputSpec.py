from nipype.interfaces.base import (
import os
class DWIRicianLMMSEFilterOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Output DWI volume.', exists=True)