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
class Evaluator:

    def __init__(self):
        self.first_call = True

    def __call__(self, controldir):
        if not self.first_call:
            try:
                repository = controldir.open_repository()
            except errors.NoRepositoryPresent:
                pass
            else:
                return (False, ([], repository))
        self.first_call = False
        value = (controldir.list_branches(), None)
        return (True, value)