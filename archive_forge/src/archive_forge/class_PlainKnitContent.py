import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
class PlainKnitContent(KnitContent):
    """Unannotated content.

    When annotate[_iter] is called on this content, the same version is reported
    for all lines. Generally, annotate[_iter] is not useful on PlainKnitContent
    objects.
    """

    def __init__(self, lines, version_id):
        KnitContent.__init__(self)
        self._lines = lines
        self._version_id = version_id

    def annotate(self):
        """Return a list of (origin, text) for each content line."""
        return [(self._version_id, line) for line in self._lines]

    def apply_delta(self, delta, new_version_id):
        """Apply delta to this object to become new_version_id."""
        offset = 0
        lines = self._lines
        for start, end, count, delta_lines in delta:
            lines[offset + start:offset + end] = delta_lines
            offset = offset + (start - end) + count
        self._version_id = new_version_id

    def copy(self):
        return PlainKnitContent(self._lines[:], self._version_id)

    def text(self):
        lines = self._lines
        if self._should_strip_eol:
            lines = lines[:]
            lines[-1] = lines[-1].rstrip(b'\n')
        return lines