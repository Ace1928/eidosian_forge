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
def _slice_match_level(self, overlay_items):
    """
        Find the match strength for a list of overlay items that must
        be exactly the same length as the pattern specification.
        """
    level = 0
    for spec, el in zip(self._pattern_spec, overlay_items):
        if spec[0] != type(el).__name__:
            return None
        level += 1
        if len(spec) == 1:
            continue
        group = [el.group, group_sanitizer(el.group, escape=False)]
        if spec[1] in group:
            level += 1
        else:
            return None
        if len(spec) == 3:
            group = [el.label, label_sanitizer(el.label, escape=False)]
            if spec[2] in group:
                level += 1
            else:
                return None
    return level