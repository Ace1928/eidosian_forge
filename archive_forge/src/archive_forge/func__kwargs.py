from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
@property
def _kwargs(self):
    return {'compression': self.filter_id, 'compression_opts': self.filter_options}