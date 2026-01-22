import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class XFibres5InputSpec(FSLXCommandInputSpec):
    gradnonlin = File(exists=True, argstr='--gradnonlin=%s', desc='gradient file corresponding to slice')