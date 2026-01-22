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
class FramewiseDisplacementOutputSpec(TraitedSpec):
    out_file = File(desc='calculated FD per timestep')
    out_figure = File(desc='output image file')
    fd_average = traits.Float(desc='average FD')