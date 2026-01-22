import itertools
from typing import Callable, Dict, Tuple, Optional
from dulwich.errors import NotCommitError
from dulwich.objects import ObjectID
from dulwich.object_store import ObjectStoreGraphWalker
from dulwich.pack import PACK_SPOOL_FILE_MAX_SIZE
from dulwich.protocol import CAPABILITY_THIN_PACK, ZERO_SHA
from dulwich.refs import SYMREF
from dulwich.walk import Walker
from .. import config, trace, ui
from ..errors import (DivergedBranches, FetchLimitUnsupported,
from ..repository import FetchResult, InterRepository, AbstractSearchResult
from ..revision import NULL_REVISION, RevisionID
from .errors import NoPushSupport
from .fetch import DetermineWantsRecorder, import_git_objects
from .mapping import needs_roundtripping
from .object_store import get_object_store
from .push import MissingObjectsIterator, remote_divergence
from .refs import is_tag, ref_to_tag_name
from .remote import RemoteGitError, RemoteGitRepository
from .repository import GitRepository, GitRepositoryFormat, LocalGitRepository
from .unpeel_map import UnpeelMap
class InterToLocalGitRepository(InterToGitRepository):
    """InterBranch implementation between a Bazaar and a Git repository."""
    target: LocalGitRepository

    def __init__(self, source, target):
        super().__init__(source, target)
        self.target_store = self.target.controldir._git.object_store
        self.target_refs = self.target.controldir._git.refs

    def _commit_needs_fetching(self, sha_id):
        try:
            return sha_id not in self.target_store
        except NoSuchRevision:
            return False

    def _revision_needs_fetching(self, sha_id, revid):
        if revid == NULL_REVISION:
            return False
        if sha_id is None:
            try:
                sha_id = self.source_store._lookup_revision_sha1(revid)
            except KeyError:
                return False
        return self._commit_needs_fetching(sha_id)

    def missing_revisions(self, stop_revisions):
        """Find the revisions that are missing from the target repository.

        :param stop_revisions: Revisions to check for (tuples with
            Git SHA1, bzr revid)
        :return: sequence of missing revisions, in topological order
        :raise: NoSuchRevision if the stop_revisions are not present in
            the source
        """
        revid_sha_map = {}
        stop_revids = []
        for sha1, revid in stop_revisions:
            if sha1 is not None and revid is not None:
                revid_sha_map[revid] = sha1
                stop_revids.append(revid)
            elif sha1 is not None:
                if self._commit_needs_fetching(sha1):
                    for kind, (revid, tree_sha, verifiers) in self.source_store.lookup_git_sha(sha1):
                        revid_sha_map[revid] = sha1
                        stop_revids.append(revid)
            else:
                if revid is None:
                    raise AssertionError
                stop_revids.append(revid)
        missing = set()
        graph = self.source.get_graph()
        with ui.ui_factory.nested_progress_bar() as pb:
            while stop_revids:
                new_stop_revids = []
                for revid in stop_revids:
                    sha1 = revid_sha_map.get(revid)
                    if revid not in missing and self._revision_needs_fetching(sha1, revid):
                        missing.add(revid)
                        new_stop_revids.append(revid)
                stop_revids = set()
                parent_map = graph.get_parent_map(new_stop_revids)
                for parent_revids in parent_map.values():
                    stop_revids.update(parent_revids)
                pb.update('determining revisions to fetch', len(missing))
        return graph.iter_topo_order(missing)

    def _get_target_either_refs(self) -> EitherRefDict:
        """Return a dictionary with references.

        :return: Dictionary with reference names as keys and tuples
            with Git SHA, Bazaar revid as values.
        """
        bzr_refs = {}
        for k in self.target._git.refs.allkeys():
            try:
                v = self.target._git.refs.read_ref(k)
            except KeyError:
                continue
            revid = None
            if v and (not v.startswith(SYMREF)):
                try:
                    for kind, type_data in self.source_store.lookup_git_sha(v):
                        if kind == 'commit' and self.source.has_revision(type_data[0]):
                            revid = type_data[0]
                            break
                except KeyError:
                    pass
            bzr_refs[k] = (v, revid)
        return bzr_refs

    def fetch_refs(self, update_refs, lossy, overwrite: bool=False):
        self._warn_slow()
        result_refs = {}
        with self.source_store.lock_read():
            old_refs = self._get_target_either_refs()
            new_refs = update_refs(old_refs)
            revidmap = self.fetch_revs([(git_sha, bzr_revid) for git_sha, bzr_revid in new_refs.values() if git_sha is None or not git_sha.startswith(SYMREF)], lossy=lossy)
            for name, (gitid, revid) in new_refs.items():
                if gitid is None:
                    try:
                        gitid = revidmap[revid][0]
                    except KeyError:
                        gitid = self.source_store._lookup_revision_sha1(revid)
                if gitid.startswith(SYMREF):
                    self.target_refs.set_symbolic_ref(name, gitid[len(SYMREF):])
                else:
                    try:
                        old_git_id = old_refs[name][0]
                    except KeyError:
                        self.target_refs.add_if_new(name, gitid)
                    else:
                        self.target_refs.set_if_equals(name, old_git_id, gitid)
                    result_refs[name] = (gitid, revid if not lossy else self.mapping.revision_id_foreign_to_bzr(gitid))
        return (revidmap, old_refs, result_refs)

    def fetch_revs(self, revs, lossy: bool, limit: Optional[int]=None) -> RevidMap:
        if not lossy and (not self.mapping.roundtripping):
            for git_sha, bzr_revid in revs:
                if bzr_revid is not None and needs_roundtripping(self.source, bzr_revid):
                    raise NoPushSupport(self.source, self.target, self.mapping, bzr_revid)
        with self.source_store.lock_read():
            todo = list(self.missing_revisions(revs))[:limit]
            revidmap = {}
            with ui.ui_factory.nested_progress_bar() as pb:
                object_generator = MissingObjectsIterator(self.source_store, self.source, pb)
                for old_revid, git_sha in object_generator.import_revisions(todo, lossy=lossy):
                    if lossy:
                        new_revid = self.mapping.revision_id_foreign_to_bzr(git_sha)
                    else:
                        new_revid = old_revid
                        try:
                            self.mapping.revision_id_bzr_to_foreign(old_revid)
                        except InvalidRevisionId:
                            pass
                    revidmap[old_revid] = (git_sha, new_revid)
                self.target_store.add_objects(object_generator)
                return revidmap

    def fetch(self, revision_id=None, find_ghosts: bool=False, lossy=False, fetch_spec=None) -> FetchResult:
        if revision_id is not None:
            stop_revisions = [(None, revision_id)]
        elif fetch_spec is not None:
            recipe = fetch_spec.get_recipe()
            if recipe[0] in ('search', 'proxy-search'):
                stop_revisions = [(None, revid) for revid in recipe[1]]
            else:
                raise AssertionError('Unsupported search result type %s' % recipe[0])
        else:
            stop_revisions = [(None, revid) for revid in self.source.all_revision_ids()]
        self._warn_slow()
        try:
            revidmap = self.fetch_revs(stop_revisions, lossy=lossy)
        except NoPushSupport:
            raise NoRoundtrippingSupport(self.source, self.target)
        return FetchResult(revidmap)

    @staticmethod
    def is_compatible(source, target):
        """Be compatible with GitRepository."""
        return not isinstance(source, GitRepository) and isinstance(target, LocalGitRepository)