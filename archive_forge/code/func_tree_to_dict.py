import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
@classmethod
def tree_to_dict(cls, tree):
    """
        Given an OptionTree, convert it into the equivalent dictionary format.
        """
    specs = {}
    for k in tree.keys():
        spec_key = '.'.join(k)
        specs[spec_key] = {}
        for grp in tree[k].groups:
            kwargs = tree[k].groups[grp].kwargs
            if kwargs:
                specs[spec_key][grp] = kwargs
    return specs