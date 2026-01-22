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
def _extend_partial_history(self, stop_index: Optional[int]=None, stop_revision: Optional[RevisionID]=None) -> None:
    """Extend the partial history to include a given index

        If a stop_index is supplied, stop when that index has been reached.
        If a stop_revision is supplied, stop when that revision is
        encountered.  Otherwise, stop when the beginning of history is
        reached.

        Args:
          stop_index: The index which should be present.  When it is
            present, history extension will stop.
          stop_revision: The revision id which should be present.  When
            it is encountered, history extension will stop.
        """
    if len(self._partial_revision_history_cache) == 0:
        self._partial_revision_history_cache = [self.last_revision()]
    repository._iter_for_revno(self.repository, self._partial_revision_history_cache, stop_index=stop_index, stop_revision=stop_revision)
    if self._partial_revision_history_cache[-1] == _mod_revision.NULL_REVISION:
        self._partial_revision_history_cache.pop()