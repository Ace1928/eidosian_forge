import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
@classmethod
def _unpack_paths(cls, objs, items, counts):
    """
        Recursively unpacks lists and ViewableTree-like objects, accumulating
        into the supplied list of items.
        """
    if type(objs) is cls:
        objs = objs.items()
    for item in objs:
        path, obj = item if isinstance(item, tuple) else (None, item)
        if type(obj) is cls:
            cls._unpack_paths(obj, items, counts)
            continue
        new = path is None or len(path) == 1
        path = util.get_path(item) if new else path
        new_path = util.make_path_unique(path, counts, new)
        items.append((new_path, obj))