import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class GenerateDirectionsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='directions file')