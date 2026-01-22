from nipype.interfaces.base import (
import os
class DWIToDTIEstimationOutputSpec(TraitedSpec):
    outputTensor = File(position=-2, desc='Estimated DTI volume', exists=True)
    outputBaseline = File(position=-1, desc='Estimated baseline volume', exists=True)