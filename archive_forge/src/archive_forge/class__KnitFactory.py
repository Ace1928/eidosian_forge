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
class _KnitFactory:
    """Base class for common Factory functions."""

    def parse_record(self, version_id, record, record_details, base_content, copy_base_content=True):
        """Parse a record into a full content object.

        :param version_id: The official version id for this content
        :param record: The data returned by read_records_iter()
        :param record_details: Details about the record returned by
            get_build_details
        :param base_content: If get_build_details returns a compression_parent,
            you must return a base_content here, else use None
        :param copy_base_content: When building from the base_content, decide
            you can either copy it and return a new object, or modify it in
            place.
        :return: (content, delta) A Content object and possibly a line-delta,
            delta may be None
        """
        method, noeol = record_details
        if method == 'line-delta':
            if copy_base_content:
                content = base_content.copy()
            else:
                content = base_content
            delta = self.parse_line_delta(record, version_id)
            content.apply_delta(delta, version_id)
        else:
            content = self.parse_fulltext(record, version_id)
            delta = None
        content._should_strip_eol = noeol
        return (content, delta)