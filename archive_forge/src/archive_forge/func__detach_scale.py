import weakref
from collections import OrderedDict
from collections.abc import MutableMapping
import h5py
import numpy as np
def _detach_scale(self):
    """Detach dimension scale from all references"""
    refs = self._scale_refs
    if refs:
        for var, dim in refs:
            self._parent._all_h5groups[var].dims[dim].detach_scale(self._h5ds)