import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SphereInputSpec(FSTraitedSpecOpenMP):
    in_file = File(argstr='%s', position=-2, copyfile=True, mandatory=True, exists=True, desc='Input file for Sphere')
    out_file = File(argstr='%s', position=-1, exists=False, name_source=['in_file'], hash_files=False, name_template='%s.sphere', desc='Output file for Sphere')
    seed = traits.Int(argstr='-seed %d', desc='Seed for setting random number generator')
    magic = traits.Bool(argstr='-q', desc='No documentation. Direct questions to analysis-bugs@nmr.mgh.harvard.edu')
    in_smoothwm = File(exists=True, copyfile=True, desc='Input surface required when -q flag is not selected')