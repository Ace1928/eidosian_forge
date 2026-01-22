import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
class EverythingResult(AbstractSearchResult):
    """A search result that simply requests everything in the repository."""

    def __init__(self, repo):
        self._repo = repo

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._repo)

    def get_recipe(self):
        raise NotImplementedError(self.get_recipe)

    def get_network_struct(self):
        return (b'everything',)

    def get_keys(self):
        if 'evil' in debug.debug_flags:
            from . import remote
            if isinstance(self._repo, remote.RemoteRepository):
                trace.mutter_callsite(2, 'EverythingResult(RemoteRepository).get_keys() is slow.')
        return self._repo.all_revision_ids()

    def is_empty(self):
        return False

    def refine(self, seen, referenced):
        heads = set(self._repo.all_revision_ids())
        heads.difference_update(seen)
        heads.update(referenced)
        return PendingAncestryResult(heads, self._repo)