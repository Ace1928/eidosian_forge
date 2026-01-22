import os
from ..base import (
class JistLaminarProfileCalculatorOutputSpec(TraitedSpec):
    outResult = File(desc='Result', exists=True)