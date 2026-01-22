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
def _get_writer_regexp_highlighted(self, path, revno, cache_id):
    """Get function for writing output with regexp match highlighted"""
    _line_writer = self._get_writer_plain(path, revno, cache_id)
    sub, highlight = (self._sub, self._highlight)

    def _line_writer_regexp_highlighted(line, **kwargs):
        """Write formatted line with matched pattern highlighted"""
        return _line_writer(line=sub(highlight, line), **kwargs)
    return _line_writer_regexp_highlighted