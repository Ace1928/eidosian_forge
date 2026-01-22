import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class DistanceMapOutputSpec(TraitedSpec):
    distance_map = File(exists=True, desc='value is distance to nearest nonzero voxels')
    local_max_file = File(desc='image of local maxima')