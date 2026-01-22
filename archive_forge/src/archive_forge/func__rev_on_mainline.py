import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def _rev_on_mainline(rev_tuple):
    """returns True is rev tuple is on mainline"""
    if len(rev_tuple) == 1:
        return True
    return rev_tuple[1] == 0 and rev_tuple[2] == 0