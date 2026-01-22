from nipype.interfaces.base import (
import os
class DTIimportOutputSpec(TraitedSpec):
    outputTensor = File(position=-1, desc='Output DTI volume', exists=True)