import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
def _set_fapl_mpio(plist, **kwargs):
    """Set file access property list for mpio driver"""
    if not mpi:
        raise ValueError("h5py was built without MPI support, can't use mpio driver")
    import mpi4py.MPI
    kwargs.setdefault('info', mpi4py.MPI.Info())
    plist.set_fapl_mpio(**kwargs)