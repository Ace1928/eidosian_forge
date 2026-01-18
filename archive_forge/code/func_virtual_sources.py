import posixpath as pp
import sys
import numpy
from .. import h5, h5s, h5t, h5r, h5d, h5p, h5fd, h5ds, _selector
from .base import (
from . import filters
from . import selections as sel
from . import selections2 as sel2
from .datatype import Datatype
from .compat import filename_decode
from .vds import VDSmap, vds_support
@with_phil
def virtual_sources(self):
    """Get a list of the data mappings for a virtual dataset"""
    if not self.is_virtual:
        raise RuntimeError('Not a virtual dataset')
    dcpl = self._dcpl
    return [VDSmap(dcpl.get_virtual_vspace(j), dcpl.get_virtual_filename(j), dcpl.get_virtual_dsetname(j), dcpl.get_virtual_srcspace(j)) for j in range(dcpl.get_virtual_count())]