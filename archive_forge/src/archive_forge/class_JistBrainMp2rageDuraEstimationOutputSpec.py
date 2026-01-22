import os
from ..base import (
class JistBrainMp2rageDuraEstimationOutputSpec(TraitedSpec):
    outDura = File(desc='Dura Image', exists=True)