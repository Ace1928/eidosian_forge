import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class CreateWarpedOutputSpec(TraitedSpec):
    warped_files = traits.List(File(exists=True, desc='final warped files'))