import os
from .base import (
from ..utils.filemanip import fname_presuffix
from ..external.due import BibTeX
class PETPVCOutputSpec(TraitedSpec):
    out_file = File(desc='Output file')