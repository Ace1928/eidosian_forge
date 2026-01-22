import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class RemoveIntersectionInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', position=-2, mandatory=True, exists=True, copyfile=True, desc='Input file for RemoveIntersection')
    out_file = File(argstr='%s', position=-1, exists=False, name_source=['in_file'], name_template='%s', hash_files=False, keep_extension=True, desc='Output file for RemoveIntersection')