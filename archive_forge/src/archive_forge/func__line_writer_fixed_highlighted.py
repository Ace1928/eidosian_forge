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
def _line_writer_fixed_highlighted(line, **kwargs):
    """Write formatted line with string searched for highlighted"""
    return _line_writer(line=line.replace(old, new), **kwargs)