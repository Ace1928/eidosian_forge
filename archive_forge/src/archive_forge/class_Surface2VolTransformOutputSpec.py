import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Surface2VolTransformOutputSpec(TraitedSpec):
    transformed_file = File(exists=True, desc='Path to output file if used normally')
    vertexvol_file = File(desc='vertex map volume path id. Optional')