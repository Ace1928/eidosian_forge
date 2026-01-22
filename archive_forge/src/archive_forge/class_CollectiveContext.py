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
class CollectiveContext:
    """ Manages collective I/O in MPI mode """

    def __init__(self, dset):
        self._dset = dset

    def __enter__(self):
        self._dset._dxpl.set_dxpl_mpio(h5fd.MPIO_COLLECTIVE)

    def __exit__(self, *args):
        self._dset._dxpl.set_dxpl_mpio(h5fd.MPIO_INDEPENDENT)