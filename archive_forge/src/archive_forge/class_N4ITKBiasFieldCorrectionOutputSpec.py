from nipype.interfaces.base import (
import os
class N4ITKBiasFieldCorrectionOutputSpec(TraitedSpec):
    outputimage = File(desc='Result of processing', exists=True)
    outputbiasfield = File(desc='Recovered bias field (OPTIONAL)', exists=True)