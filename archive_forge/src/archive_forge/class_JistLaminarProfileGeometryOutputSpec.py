import os
from ..base import (
class JistLaminarProfileGeometryOutputSpec(TraitedSpec):
    outResult = File(desc='Result', exists=True)