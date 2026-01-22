import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
class PendingAncestryResult(AbstractSearchResult):
    """A search result that will reconstruct the ancestry for some graph heads.

    Unlike SearchResult, this doesn't hold the complete search result in
    memory, it just holds a description of how to generate it.
    """

    def __init__(self, heads, repo):
        """Constructor.

        :param heads: an iterable of graph heads.
        :param repo: a repository to use to generate the ancestry for the given
            heads.
        """
        self.heads = frozenset(heads)
        self.repo = repo

    def __repr__(self):
        if len(self.heads) > 5:
            heads_repr = repr(list(self.heads)[:5])[:-1]
            heads_repr += ', <%d more>...]' % (len(self.heads) - 5,)
        else:
            heads_repr = repr(self.heads)
        return '<{} heads:{} repo:{!r}>'.format(self.__class__.__name__, heads_repr, self.repo)

    def get_recipe(self):
        """Return a recipe that can be used to replay this search.

        The recipe allows reconstruction of the same results at a later date.

        :seealso SearchResult.get_recipe:

        :return: A tuple ('proxy-search', start_keys_set, set(), -1)
            To recreate this result, create a PendingAncestryResult with the
            start_keys_set.
        """
        return ('proxy-search', self.heads, set(), -1)

    def get_network_struct(self):
        parts = [b'ancestry-of']
        parts.extend(self.heads)
        return parts

    def get_keys(self):
        """See SearchResult.get_keys.

        Returns all the keys for the ancestry of the heads, excluding
        NULL_REVISION.
        """
        return self._get_keys(self.repo.get_graph())

    def _get_keys(self, graph):
        NULL_REVISION = revision.NULL_REVISION
        keys = [key for key, parents in graph.iter_ancestry(self.heads) if key != NULL_REVISION and parents is not None]
        return keys

    def is_empty(self):
        """Return false if the search lists 1 or more revisions."""
        if revision.NULL_REVISION in self.heads:
            return len(self.heads) == 1
        else:
            return len(self.heads) == 0

    def refine(self, seen, referenced):
        """Create a new search by refining this search.

        :param seen: Revisions that have been satisfied.
        :param referenced: Revision references observed while satisfying some
            of this search.
        """
        referenced = self.heads.union(referenced)
        return PendingAncestryResult(referenced - seen, self.repo)