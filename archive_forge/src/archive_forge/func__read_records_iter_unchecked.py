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
def _read_records_iter_unchecked(self, records):
    """Read text records from data file and yield raw data.

        No validation is done.

        Yields tuples of (key, data).
        """
    if len(records):
        needed_offsets = [index_memo for key, index_memo in records]
        raw_records = self._access.get_raw_records(needed_offsets)
    for key, index_memo in records:
        data = next(raw_records)
        yield (key, data)