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
class FramewiseDisplacementInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='motion parameters')
    parameter_source = traits.Enum('FSL', 'AFNI', 'SPM', 'FSFAST', 'NIPY', desc='Source of movement parameters', mandatory=True)
    radius = traits.Float(50, usedefault=True, desc='radius in mm to calculate angular FDs, 50mm is the default since it is used in Power et al. 2012')
    out_file = File('fd_power_2012.txt', usedefault=True, desc='output file name')
    out_figure = File('fd_power_2012.pdf', usedefault=True, desc='output figure name')
    series_tr = traits.Float(desc='repetition time in sec.')
    save_plot = traits.Bool(False, usedefault=True, desc='write FD plot')
    normalize = traits.Bool(False, usedefault=True, desc='calculate FD in mm/s')
    figdpi = traits.Int(100, usedefault=True, desc='output dpi for the FD plot')
    figsize = traits.Tuple(traits.Float(11.7), traits.Float(2.3), usedefault=True, desc='output figure size')