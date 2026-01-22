import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class RemoveNeckInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', exists=True, mandatory=True, position=-4, desc='Input file for RemoveNeck')
    out_file = File(argstr='%s', exists=False, name_source=['in_file'], name_template='%s_noneck', hash_files=False, keep_extension=True, position=-1, desc='Output file for RemoveNeck')
    transform = File(argstr='%s', exists=True, mandatory=True, position=-3, desc='Input transform file for RemoveNeck')
    template = File(argstr='%s', exists=True, mandatory=True, position=-2, desc='Input template file for RemoveNeck')
    radius = traits.Int(argstr='-radius %d', desc='Radius')