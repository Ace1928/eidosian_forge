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
class ACompCor(CompCor):
    """
    Anatomical compcor: for inputs and outputs, see CompCor.
    When the mask provided is an anatomical mask, then CompCor
    is equivalent to ACompCor.
    """

    def __init__(self, *args, **kwargs):
        """exactly the same as compcor except the header"""
        super(ACompCor, self).__init__(*args, **kwargs)
        self._header = 'aCompCor'