import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class ApplyDeformationFieldOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(File(exists=True))