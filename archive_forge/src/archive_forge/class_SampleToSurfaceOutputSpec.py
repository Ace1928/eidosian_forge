import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SampleToSurfaceOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='surface file')
    hits_file = File(exists=True, desc='image with number of hits at each voxel')
    vox_file = File(exists=True, desc='text file with the number of voxels intersecting the surface')