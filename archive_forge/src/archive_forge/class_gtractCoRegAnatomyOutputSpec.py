import os
from ...base import (
class gtractCoRegAnatomyOutputSpec(TraitedSpec):
    outputTransformName = File(desc='Required: filename for the  fit transform.', exists=True)