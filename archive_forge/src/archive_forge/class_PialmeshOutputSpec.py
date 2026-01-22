import os
import re as regex
from ..base import (
class PialmeshOutputSpec(TraitedSpec):
    outputSurfaceFile = File(desc='path/name of surface file')