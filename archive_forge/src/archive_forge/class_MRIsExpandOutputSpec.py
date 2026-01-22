import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsExpandOutputSpec(TraitedSpec):
    out_file = File(desc='Output surface file')