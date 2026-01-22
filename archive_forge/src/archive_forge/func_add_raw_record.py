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
def add_raw_record(self, key, size, raw_data):
    """Add raw knit bytes to a storage area.

        The data is spooled to the container writer in one bytes-record per
        raw data item.

        :param key: The key of the raw data segment
        :param size: The size of the raw data segment
        :param raw_data: A chunked bytestring containing the data.
        :return: opaque index memo to retrieve the record later.
            For _KnitKeyAccess the memo is (key, pos, length), where the key is
            the record key.
        """
    path = self._mapper.map(key)
    try:
        base = self._transport.append_bytes(path + '.knit', b''.join(raw_data))
    except _mod_transport.NoSuchFile:
        self._transport.mkdir(osutils.dirname(path))
        base = self._transport.append_bytes(path + '.knit', b''.join(raw_data))
    return (key, base, size)