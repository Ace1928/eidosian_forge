import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MakeDyadicVectorsOutputSpec(TraitedSpec):
    dyads = File(exists=True)
    dispersion = File(exists=True)