import os
from ...base import (
class gtractCreateGuideFiberOutputSpec(TraitedSpec):
    outputFiber = File(desc='Required: output guide fiber file name', exists=True)