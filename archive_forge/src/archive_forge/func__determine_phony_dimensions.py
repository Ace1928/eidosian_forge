import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import h5py
import numpy as np
from packaging import version
from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen
def _determine_phony_dimensions(self):

    def create_phony_dimensions(grp):
        for name in grp.groups:
            create_phony_dimensions(grp[name])
    create_phony_dimensions(self)