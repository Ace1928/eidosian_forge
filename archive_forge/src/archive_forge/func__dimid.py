import weakref
from collections import OrderedDict
from collections.abc import MutableMapping
import h5py
import numpy as np
@property
def _dimid(self):
    if self._phony:
        return False
    return self._h5ds.attrs.get('_Netcdf4Dimid', self._dimensionid)