from nipype.interfaces.base import (
import os
class CurvatureAnisotropicDiffusionOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Output filtered', exists=True)