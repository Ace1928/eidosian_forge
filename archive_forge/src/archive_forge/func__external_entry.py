from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
def _external_entry(entry):
    """ Check for and return a well-formed entry tuple for
    a call to h5p.set_external. """
    if not isinstance(entry, tuple):
        raise TypeError('Each external entry must be a tuple of (name, offset, size)')
    name, offset, size = entry
    name = filename_encode(name)
    offset = operator.index(offset)
    size = operator.index(size)
    return (name, offset, size)