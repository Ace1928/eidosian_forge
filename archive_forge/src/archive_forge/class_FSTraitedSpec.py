import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
class FSTraitedSpec(CommandLineInputSpec):
    subjects_dir = Directory(exists=True, desc='subjects directory')