from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def _default_transpose(x, axes):
    return x.transpose(axes)