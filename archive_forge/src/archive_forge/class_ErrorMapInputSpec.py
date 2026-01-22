import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class ErrorMapInputSpec(BaseInterfaceInputSpec):
    in_ref = File(exists=True, mandatory=True, desc='Reference image. Requires the same dimensions as in_tst.')
    in_tst = File(exists=True, mandatory=True, desc='Test image. Requires the same dimensions as in_ref.')
    mask = File(exists=True, desc='calculate overlap only within this mask.')
    metric = traits.Enum('sqeuclidean', 'euclidean', desc='error map metric (as implemented in scipy cdist)', usedefault=True, mandatory=True)
    out_map = File(desc='Name for the output file')