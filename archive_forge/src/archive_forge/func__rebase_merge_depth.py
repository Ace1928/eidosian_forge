import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def _rebase_merge_depth(view_revisions):
    """Adjust depths upwards so the top level is 0."""
    if view_revisions and view_revisions[0][2] and view_revisions[-1][2]:
        min_depth = min([d for r, n, d in view_revisions])
        if min_depth != 0:
            view_revisions = [(r, n, d - min_depth) for r, n, d in view_revisions]
    return view_revisions