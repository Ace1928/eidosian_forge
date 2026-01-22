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
class InterRepository(InterObject[Repository]):
    """This class represents operations taking place between two repositories.

    Its instances have methods like copy_content and fetch, and contain
    references to the source and target repositories these operations can be
    carried out on.

    Often we will provide convenience methods on 'repository' which carry out
    operations with another repository - they will always forward to
    InterRepository.get(other).method_name(parameters).
    """
    _optimisers = []
    'The available optimised InterRepository types.'

    def copy_content(self, revision_id: Optional[_mod_revision.RevisionID]=None) -> None:
        """Make a complete copy of the content in self into destination.

        This is a destructive operation! Do not use it on existing
        repositories.

        Args:
          revision_id: Only copy the content needed to construct
                            revision_id and its parents.
        """
        with self.lock_write():
            try:
                self.target.set_make_working_trees(self.source.make_working_trees())
            except (NotImplementedError, errors.RepositoryUpgradeRequired):
                pass
            self.target.fetch(self.source, revision_id=revision_id)

    def fetch(self, revision_id: Optional[_mod_revision.RevisionID]=None, find_ghosts: bool=False, lossy: bool=False) -> FetchResult:
        """Fetch the content required to construct revision_id.

        The content is copied from self.source to self.target.

        Args:
          revision_id: if None all content is copied, if NULL_REVISION no
                            content is copied.
        Returns: FetchResult
        """
        raise NotImplementedError(self.fetch)

    def search_missing_revision_ids(self, find_ghosts: bool=True, revision_ids: Optional[Iterable[_mod_revision.RevisionID]]=None, if_present_ids: Optional[Iterable[_mod_revision.RevisionID]]=None, limit: Optional[int]=None) -> AbstractSearchResult:
        """Return the revision ids that source has that target does not.

        Args:
          revision_ids: return revision ids included by these
            revision_ids.  NoSuchRevision will be raised if any of these
            revisions are not present.
          if_present_ids: like revision_ids, but will not cause
            NoSuchRevision if any of these are absent, instead they will simply
            not be in the result.  This is useful for e.g. finding revisions
            to fetch for tags, which may reference absent revisions.
          find_ghosts: If True find missing revisions in deep history
            rather than just finding the surface difference.
          limit: Maximum number of revisions to return, topologically
            ordered
        Returns: A SearchResult.
        """
        raise NotImplementedError(self.search_missing_revision_ids)

    @staticmethod
    def _same_model(source, target):
        """True if source and target have the same data representation.

        Note: this is always called on the base class; overriding it in a
        subclass will have no effect.
        """
        try:
            InterRepository._assert_same_model(source, target)
            return True
        except errors.IncompatibleRepositories as e:
            return False

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