from nipype.interfaces.base import (
import os
class ExpertAutomatedRegistrationOutputSpec(TraitedSpec):
    resampledImage = File(desc='Registration results', exists=True)
    saveTransform = File(desc='Save the transform that results from registration', exists=True)