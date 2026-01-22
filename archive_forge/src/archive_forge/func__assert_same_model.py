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
@staticmethod
def _assert_same_model(source, target):
    """Raise an exception if two repositories do not use the same model.
        """
    if source.supports_rich_root() != target.supports_rich_root():
        raise errors.IncompatibleRepositories(source, target, 'different rich-root support')
    if not hasattr(source, '_serializer') or not hasattr(target, '_serializer'):
        if source != target:
            raise errors.IncompatibleRepositories(source, target, 'different formats')
        return
    if source._serializer != target._serializer:
        raise errors.IncompatibleRepositories(source, target, 'different serializers')