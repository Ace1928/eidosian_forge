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
class RepositoryFormatRegistry(controldir.ControlComponentFormatRegistry):
    """Repository format registry."""

    def get_default(self):
        """Return the current default format."""
        return controldir.format_registry.make_controldir('default').repository_format