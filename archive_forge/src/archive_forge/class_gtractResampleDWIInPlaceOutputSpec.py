import os
from ...base import (
class gtractResampleDWIInPlaceOutputSpec(TraitedSpec):
    outputResampledB0 = File(desc='Convenience function for extracting the first index location (assumed to be the B0)', exists=True)
    outputVolume = File(desc='Required: output image (NRRD file) that has been rigidly transformed into the space of the structural image and padded if image padding was changed from 0,0,0 default.', exists=True)