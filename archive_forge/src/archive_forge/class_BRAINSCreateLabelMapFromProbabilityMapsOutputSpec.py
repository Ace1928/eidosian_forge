import os
from ...base import (
class BRAINSCreateLabelMapFromProbabilityMapsOutputSpec(TraitedSpec):
    dirtyLabelVolume = File(desc='the labels prior to cleaning', exists=True)
    cleanLabelVolume = File(desc='the foreground labels volume', exists=True)