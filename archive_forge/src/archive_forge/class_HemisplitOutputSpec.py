import os
import re as regex
from ..base import (
class HemisplitOutputSpec(TraitedSpec):
    outputLeftHemisphere = File(desc='path/name of left hemisphere')
    outputRightHemisphere = File(desc='path/name of right hemisphere')
    outputLeftPialHemisphere = File(desc='path/name of left pial hemisphere')
    outputRightPialHemisphere = File(desc='path/name of right pial hemisphere')