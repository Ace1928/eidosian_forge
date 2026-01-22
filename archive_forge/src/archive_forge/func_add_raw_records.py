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
def add_raw_records(self, key_sizes, raw_data):
    """Add raw knit bytes to a storage area.

        The data is spooled to the container writer in one bytes-record per
        raw data item.

        :param sizes: An iterable of tuples containing the key and size of each
            raw data segment.
        :param raw_data: A chunked bytestring containing the data.
        :return: A list of memos to retrieve the record later. Each memo is an
            opaque index memo. For _KnitKeyAccess the memo is (key, pos,
            length), where the key is the record key.
        """
    raw_data = b''.join(raw_data)
    if not isinstance(raw_data, bytes):
        raise AssertionError('data must be plain bytes was %s' % type(raw_data))
    result = []
    offset = 0
    for key, size in key_sizes:
        record_bytes = [raw_data[offset:offset + size]]
        result.append(self.add_raw_record(key, size, record_bytes))
        offset += size
    return result