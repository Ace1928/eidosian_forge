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
def _get_record_map(self, keys, allow_missing=False):
    """Produce a dictionary of knit records.

        :return: {key:(record, record_details, digest, next)}

            * record: data returned from read_records (a KnitContentobject)
            * record_details: opaque information to pass to parse_record
            * digest: SHA1 digest of the full text after all steps are done
            * next: build-parent of the version, i.e. the leftmost ancestor.
                Will be None if the record is not a delta.

        :param keys: The keys to build a map for
        :param allow_missing: If some records are missing, rather than
            error, just return the data that could be generated.
        """
    raw_map = self._get_record_map_unparsed(keys, allow_missing=allow_missing)
    return self._raw_map_to_record_map(raw_map)