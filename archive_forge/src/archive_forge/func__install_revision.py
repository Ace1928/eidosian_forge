import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def _install_revision(self, revision_id, metadata, text):
    if self._repository.has_revision(revision_id):
        return
    revision = self._source_serializer.read_revision_from_string(text)
    self._repository.add_revision(revision.revision_id, revision)