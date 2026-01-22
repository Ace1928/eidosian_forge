import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, File, traits, isdefined
from .base import (
class CSDOutputSpec(TraitedSpec):
    model = File(desc='Python pickled object of the CSD model fitted.')
    out_fods = File(desc='fODFs output file name')