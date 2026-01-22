import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class RemoveNeckOutputSpec(TraitedSpec):
    out_file = File(exists=False, desc='Output file with neck removed')