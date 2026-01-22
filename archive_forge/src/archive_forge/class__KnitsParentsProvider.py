from typing import Type
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import transport as _mod_transport
from ..repository import InterRepository, IsInWriteGroupError, Repository
from .repository import RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (InterSameDataRepository,
class _KnitsParentsProvider:

    def __init__(self, knit, prefix=()):
        """Create a parent provider for string keys mapped to tuple keys."""
        self._knit = knit
        self._prefix = prefix

    def __repr__(self):
        return 'KnitsParentsProvider(%r)' % self._knit

    def get_parent_map(self, keys):
        """See graph.StackedParentsProvider.get_parent_map"""
        parent_map = self._knit.get_parent_map([self._prefix + (key,) for key in keys])
        result = {}
        for key, parents in parent_map.items():
            revid = key[-1]
            if len(parents) == 0:
                parents = (_mod_revision.NULL_REVISION,)
            else:
                parents = tuple((parent[-1] for parent in parents))
            result[revid] = parents
        for revision_id in keys:
            if revision_id == _mod_revision.NULL_REVISION:
                result[revision_id] = ()
        return result