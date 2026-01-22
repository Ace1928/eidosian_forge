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
def _activate_fallback_location(self, url, possible_transports):
    """Activate the branch/repository from url as a fallback repository."""
    for existing_fallback_repo in self.repository._fallback_repositories:
        if existing_fallback_repo.user_url == url:
            mutter('duplicate activation of fallback %r on %r', url, self)
            return
    repo = self._get_fallback_repository(url, possible_transports)
    if repo.has_same_location(self.repository):
        raise errors.UnstackableLocationError(self.user_url, url)
    self.repository.add_fallback_repository(repo)