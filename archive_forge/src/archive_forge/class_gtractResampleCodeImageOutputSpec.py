import os
from ...base import (
class gtractResampleCodeImageOutputSpec(TraitedSpec):
    outputVolume = File(desc='Required: name of output NRRD file containing the resampled code image in acquisition space.', exists=True)