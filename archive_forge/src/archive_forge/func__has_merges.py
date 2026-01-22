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
def _has_merges(branch, rev_id):
    """Does a revision have multiple parents or not?"""
    parents = branch.repository.get_parent_map([rev_id]).get(rev_id, [])
    return len(parents) > 1