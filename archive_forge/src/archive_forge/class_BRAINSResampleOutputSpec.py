from nipype.interfaces.base import (
import os
class BRAINSResampleOutputSpec(TraitedSpec):
    outputVolume = File(desc='Resulting deformed image', exists=True)