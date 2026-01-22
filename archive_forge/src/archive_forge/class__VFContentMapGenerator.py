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
class _VFContentMapGenerator(_ContentMapGenerator):
    """Content map generator reading from a VersionedFiles object."""

    def __init__(self, versioned_files, keys, nonlocal_keys=None, global_map=None, raw_record_map=None, ordering='unordered'):
        """Create a _ContentMapGenerator.

        :param versioned_files: The versioned files that the texts are being
            extracted from.
        :param keys: The keys to produce content maps for.
        :param nonlocal_keys: An iterable of keys(possibly intersecting keys)
            which are known to not be in this knit, but rather in one of the
            fallback knits.
        :param global_map: The result of get_parent_map(keys) (or a supermap).
            This is required if get_record_stream() is to be used.
        :param raw_record_map: A unparsed raw record map to use for answering
            contents.
        """
        _ContentMapGenerator.__init__(self, ordering=ordering)
        self.vf = versioned_files
        self.keys = list(keys)
        if nonlocal_keys is None:
            self.nonlocal_keys = set()
        else:
            self.nonlocal_keys = frozenset(nonlocal_keys)
        self.global_map = global_map
        self._text_map = {}
        self._contents_map = {}
        self._record_map = None
        if raw_record_map is None:
            self._raw_record_map = self.vf._get_record_map_unparsed(keys, allow_missing=True)
        else:
            self._raw_record_map = raw_record_map
        self._factory = self.vf._factory