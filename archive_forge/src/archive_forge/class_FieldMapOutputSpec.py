import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class FieldMapOutputSpec(TraitedSpec):
    vdm = File(exists=True, desc='voxel difference map')