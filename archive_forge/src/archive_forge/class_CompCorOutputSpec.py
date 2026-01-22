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
class CompCorOutputSpec(TraitedSpec):
    components_file = File(exists=True, desc='text file containing the noise components')
    pre_filter_file = File(desc='text file containing high-pass filter basis')
    metadata_file = File(desc='text file containing component metadata')