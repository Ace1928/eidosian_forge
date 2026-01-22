import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class EulerNumberOutputSpec(TraitedSpec):
    euler = traits.Int(desc='Euler number of cortical surface. A value of 2 signals a topologically correct surface model with no holes')
    defects = traits.Int(desc='Number of defects')