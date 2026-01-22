import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class SphericalAverageOutputSpec(TraitedSpec):
    out_file = File(exists=False, desc='Output label')