import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
def _make_headers(self, num_col):
    header = self.inputs.header_prefix if isdefined(self.inputs.header_prefix) else self._header
    headers = ['{}{:02d}'.format(header, i) for i in range(num_col)]
    return headers