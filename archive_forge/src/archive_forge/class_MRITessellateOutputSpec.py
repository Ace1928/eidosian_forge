import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRITessellateOutputSpec(TraitedSpec):
    """
    Uses Freesurfer's mri_tessellate to create surfaces by tessellating a given input volume
    """
    surface = File(exists=True, desc='binary surface of the tessellation ')