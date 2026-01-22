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
def _validate_revprops(self, revprops):
    for key, value in revprops.items():
        if not isinstance(value, str):
            raise ValueError('revision property (%s) is not a valid (unicode) string: %r' % (key, value))
        self._validate_unicode_text(value, 'revision property ({})'.format(key))