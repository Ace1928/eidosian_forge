import os
import re as regex
from ..base import (
class CortexOutputSpec(TraitedSpec):
    outputCerebrumMask = File(desc='path/name of cerebrum mask')