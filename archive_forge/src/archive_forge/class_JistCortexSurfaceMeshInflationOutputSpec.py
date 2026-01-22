import os
from ..base import (
class JistCortexSurfaceMeshInflationOutputSpec(TraitedSpec):
    outOriginal = File(desc='Original Surface', exists=True)
    outInflated = File(desc='Inflated Surface', exists=True)