from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def _do_revision_id_to_dotted_revno(self, revision_id):
    """Worker function for revision_id_to_revno."""
    result = self._partial_revision_id_to_revno_cache.get(revision_id)
    if result is not None:
        return result
    if self._revision_id_to_revno_cache:
        result = self._revision_id_to_revno_cache.get(revision_id)
        if result is None:
            raise errors.NoSuchRevision(self, revision_id)
    try:
        revno = self.revision_id_to_revno(revision_id)
        return (revno,)
    except errors.NoSuchRevision as exc:
        result = self.get_revision_id_to_revno_map().get(revision_id)
        if result is None:
            raise errors.NoSuchRevision(self, revision_id) from exc
    return result