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
def _check_ready_for_annotations(self, key, parent_keys):
    """return true if this text is ready to be yielded.

        Otherwise, this will return False, and queue the text into
        self._pending_annotation
        """
    for parent_key in parent_keys:
        if parent_key not in self._annotations_cache:
            self._pending_annotation.setdefault(parent_key, []).append((key, parent_keys))
            return False
    return True