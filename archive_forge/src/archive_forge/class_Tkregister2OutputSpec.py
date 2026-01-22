import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Tkregister2OutputSpec(TraitedSpec):
    reg_file = File(exists=True, desc='freesurfer-style registration file')
    fsl_file = File(desc='FSL-style registration file')
    lta_file = File(desc='LTA-style registration file')