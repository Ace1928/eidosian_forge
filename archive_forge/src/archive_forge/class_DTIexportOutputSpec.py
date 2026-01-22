from nipype.interfaces.base import (
import os
class DTIexportOutputSpec(TraitedSpec):
    outputFile = File(position=-1, desc='Output DTI file', exists=True)