from typing import List, Type, TYPE_CHECKING, Optional, Iterable
from .lazy_import import lazy_import
import time
from breezy import (
from breezy.i18n import gettext
from . import controldir, debug, errors, graph, registry, revision as _mod_revision, ui
from .decorators import only_raises
from .inter import InterObject
from .lock import LogicalLockResult, _RelockDebugMixin
from .revisiontree import RevisionTree
from .trace import (log_exception_quietly, mutter, mutter_callsite, note,
class RepositoryWriteLockResult(LogicalLockResult):
    """The result of write locking a repository.

    Attributes:
      repository_token: The token obtained from the underlying lock, or
        None.
      unlock: A callable which will unlock the lock.
    """

    def __init__(self, unlock, repository_token):
        LogicalLockResult.__init__(self, unlock)
        self.repository_token = repository_token

    def __repr__(self):
        return 'RepositoryWriteLockResult({}, {})'.format(self.repository_token, self.unlock)