import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class RelabelHypointensitiesInputSpec(FSTraitedSpec):
    lh_white = File(mandatory=True, exists=True, copyfile=True, desc='Implicit input file must be lh.white')
    rh_white = File(mandatory=True, exists=True, copyfile=True, desc='Implicit input file must be rh.white')
    aseg = File(argstr='%s', position=-3, mandatory=True, exists=True, desc='Input aseg file')
    surf_directory = Directory('.', argstr='%s', position=-2, exists=True, usedefault=True, desc='Directory containing lh.white and rh.white')
    out_file = File(argstr='%s', position=-1, exists=False, name_source=['aseg'], name_template='%s.hypos.mgz', hash_files=False, keep_extension=False, desc='Output aseg file')