import weakref
from collections import OrderedDict
from collections.abc import MutableMapping
import h5py
import numpy as np
@property
def _maxsize(self):
    return None if self.isunlimited() else self.size