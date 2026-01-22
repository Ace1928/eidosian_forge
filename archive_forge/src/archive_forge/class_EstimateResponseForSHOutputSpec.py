import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class EstimateResponseForSHOutputSpec(TraitedSpec):
    response = File(exists=True, desc='Spherical harmonics image')