import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
class NotInOtherForRevs(AbstractSearch):
    """Find all revisions missing in one repo for a some specific heads."""

    def __init__(self, to_repo, from_repo, required_ids, if_present_ids=None, find_ghosts=False, limit=None):
        """Constructor.

        :param required_ids: revision IDs of heads that must be found, or else
            the search will fail with NoSuchRevision.  All revisions in their
            ancestry not already in the other repository will be included in
            the search result.
        :param if_present_ids: revision IDs of heads that may be absent in the
            source repository.  If present, then their ancestry not already
            found in other will be included in the search result.
        :param limit: maximum number of revisions to fetch
        """
        self.to_repo = to_repo
        self.from_repo = from_repo
        self.find_ghosts = find_ghosts
        self.required_ids = required_ids
        self.if_present_ids = if_present_ids
        self.limit = limit

    def __repr__(self):
        if len(self.required_ids) > 5:
            reqd_revs_repr = repr(list(self.required_ids)[:5])[:-1] + ', ...]'
        else:
            reqd_revs_repr = repr(self.required_ids)
        if self.if_present_ids and len(self.if_present_ids) > 5:
            ifp_revs_repr = repr(list(self.if_present_ids)[:5])[:-1] + ', ...]'
        else:
            ifp_revs_repr = repr(self.if_present_ids)
        return "<%s from:%r to:%r find_ghosts:%r req'd:%r if-present:%rlimit:%r>" % (self.__class__.__name__, self.from_repo, self.to_repo, self.find_ghosts, reqd_revs_repr, ifp_revs_repr, self.limit)

    def execute(self):
        return self.to_repo.search_missing_revision_ids(self.from_repo, revision_ids=self.required_ids, if_present_ids=self.if_present_ids, find_ghosts=self.find_ghosts, limit=self.limit)