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
def _raw_map_to_record_map(self, raw_map):
    """Parse the contents of _get_record_map_unparsed.

        :return: see _get_record_map.
        """
    result = {}
    for key in raw_map:
        data, record_details, next = raw_map[key]
        content, digest = self._parse_record(key[-1], data)
        result[key] = (content, record_details, digest, next)
    return result