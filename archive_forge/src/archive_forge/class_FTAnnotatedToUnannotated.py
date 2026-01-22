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
class FTAnnotatedToUnannotated(KnitAdapter):
    """An adapter from FT annotated knits to unannotated ones."""

    def get_bytes(self, factory, target_storage_kind):
        if target_storage_kind != 'knit-ft-gz':
            raise UnavailableRepresentation(factory.key, target_storage_kind, factory.storage_kind)
        annotated_compressed_bytes = factory._raw_record
        rec, contents = self._data._parse_record_unchecked(annotated_compressed_bytes)
        content = self._annotate_factory.parse_fulltext(contents, rec[1])
        size, chunks = self._data._record_to_data((rec[1],), rec[3], content.text())
        return b''.join(chunks)