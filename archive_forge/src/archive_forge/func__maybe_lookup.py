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
def _maybe_lookup(self, name_or_index):
    """Convert possible symbolic name to index, or pass through indexes.

        NOT FOR PUBLIC USE.
        """
    if isinstance(name_or_index, int):
        return name_or_index
    else:
        return self._lookup(name_or_index)