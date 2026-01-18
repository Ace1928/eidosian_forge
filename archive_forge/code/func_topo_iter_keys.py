import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def topo_iter_keys(vf, keys=None):
    if keys is None:
        keys = vf.keys()
    parents = vf.get_parent_map(keys)
    return _topo_iter(parents, keys)