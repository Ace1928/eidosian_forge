import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MakeSurfacesOutputSpec(TraitedSpec):
    out_white = File(exists=False, desc='Output white matter hemisphere surface')
    out_curv = File(exists=False, desc='Output curv file for MakeSurfaces')
    out_area = File(exists=False, desc='Output area file for MakeSurfaces')
    out_cortex = File(exists=False, desc='Output cortex file for MakeSurfaces')
    out_pial = File(exists=False, desc='Output pial surface for MakeSurfaces')
    out_thickness = File(exists=False, desc='Output thickness file for MakeSurfaces')