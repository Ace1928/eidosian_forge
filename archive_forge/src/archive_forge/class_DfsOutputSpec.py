import os
import re as regex
from ..base import (
class DfsOutputSpec(TraitedSpec):
    outputSurfaceFile = File(desc='path/name of surface file')