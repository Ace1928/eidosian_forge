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
def _process_pending(self, key):
    """The content for 'key' was just processed.

        Determine if there is any more pending work to be processed.
        """
    to_return = []
    if key in self._pending_deltas:
        compression_parent = key
        children = self._pending_deltas.pop(key)
        for child_key, parent_keys, record, record_details in children:
            lines = self._expand_record(child_key, parent_keys, compression_parent, record, record_details)
            if self._check_ready_for_annotations(child_key, parent_keys):
                to_return.append(child_key)
    if key in self._pending_annotation:
        children = self._pending_annotation.pop(key)
        to_return.extend([c for c, p_keys in children if self._check_ready_for_annotations(c, p_keys)])
    return to_return