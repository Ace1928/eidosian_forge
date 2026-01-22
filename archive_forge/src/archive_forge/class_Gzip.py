from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
class Gzip(FilterRefBase):
    filter_id = h5z.FILTER_DEFLATE

    def __init__(self, level=DEFAULT_GZIP):
        self.filter_options = (level,)