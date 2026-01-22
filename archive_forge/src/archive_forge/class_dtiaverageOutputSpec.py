import os
from ...base import (
class dtiaverageOutputSpec(TraitedSpec):
    tensor_output = File(desc='Averaged tensor volume', exists=True)