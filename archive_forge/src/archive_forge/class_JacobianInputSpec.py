import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class JacobianInputSpec(FSTraitedSpec):
    in_origsurf = File(argstr='%s', position=-3, mandatory=True, exists=True, desc='Original surface')
    in_mappedsurf = File(argstr='%s', position=-2, mandatory=True, exists=True, desc='Mapped surface')
    out_file = File(argstr='%s', exists=False, position=-1, name_source=['in_origsurf'], hash_files=False, name_template='%s.jacobian', keep_extension=False, desc='Output Jacobian of the surface mapping')