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
def _run_post_repo_init_hooks(self, repository, controldir, shared):
    from .controldir import ControlDir, RepoInitHookParams
    hooks = ControlDir.hooks['post_repo_init']
    if not hooks:
        return
    params = RepoInitHookParams(repository, self, controldir, shared)
    for hook in hooks:
        hook(params)