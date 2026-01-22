import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
@staticmethod
def _lca_multi_way(bases, other, this, allow_overriding_lca=True):
    """Consider LCAs when determining whether a change has occurred.

        If LCAS are all identical, this is the same as a _three_way comparison.

        :param bases: value in (BASE, [LCAS])
        :param other: value in OTHER
        :param this: value in THIS
        :param allow_overriding_lca: If there is more than one unique lca
            value, allow OTHER to override THIS if it has a new value, and
            THIS only has an lca value, or vice versa. This is appropriate for
            truly scalar values, not as much for non-scalars.
        :return: 'this', 'other', or 'conflict' depending on whether an entry
            changed or not.
        """
    if other == this:
        return 'this'
    base_val, lca_vals = bases
    filtered_lca_vals = [lca_val for lca_val in lca_vals if lca_val != base_val]
    if len(filtered_lca_vals) == 0:
        return Merge3Merger._three_way(base_val, other, this)
    unique_lca_vals = set(filtered_lca_vals)
    if len(unique_lca_vals) == 1:
        return Merge3Merger._three_way(unique_lca_vals.pop(), other, this)
    if allow_overriding_lca:
        if other in unique_lca_vals:
            if this in unique_lca_vals:
                return 'conflict'
            else:
                return 'this'
        elif this in unique_lca_vals:
            return 'other'
    return 'conflict'