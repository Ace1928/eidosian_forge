from nipype.interfaces.base import (
import os
class MultiResolutionAffineRegistrationOutputSpec(TraitedSpec):
    resampledImage = File(desc='Registration results', exists=True)
    saveTransform = File(desc='Save the output transform from the registration', exists=True)