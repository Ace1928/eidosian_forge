import os
from copy import copy
from io import BytesIO
import patiencediff
from ..lazy_import import lazy_import
from breezy import tsort
from .. import errors, osutils
from .. import transport as _mod_transport
from ..errors import RevisionAlreadyPresent, RevisionNotPresent
from ..osutils import dirname, sha, sha_strings, split_lines
from ..revision import NULL_REVISION
from ..trace import mutter
from .versionedfile import (AbsentContentFactory, ContentFactory,
from .weavefile import _read_weave_v5, write_weave_v5
class WeaveContentFactory(ContentFactory):
    """Content factory for streaming from weaves.

    :seealso ContentFactory:
    """

    def __init__(self, version, weave):
        """Create a WeaveContentFactory for version from weave."""
        ContentFactory.__init__(self)
        self.sha1 = weave.get_sha1s([version])[version]
        self.key = (version,)
        parents = weave.get_parent_map([version])[version]
        self.parents = tuple(((parent,) for parent in parents))
        self.storage_kind = 'fulltext'
        self._weave = weave

    def get_bytes_as(self, storage_kind):
        if storage_kind == 'fulltext':
            return self._weave.get_text(self.key[-1])
        elif storage_kind in ('chunked', 'lines'):
            return self._weave.get_lines(self.key[-1])
        else:
            raise UnavailableRepresentation(self.key, storage_kind, 'fulltext')

    def iter_bytes_as(self, storage_kind):
        if storage_kind in ('chunked', 'lines'):
            return iter(self._weave.get_lines(self.key[-1]))
        else:
            raise UnavailableRepresentation(self.key, storage_kind, 'fulltext')