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
class InterGitNonGitRepository(InterFromGitRepository):
    """Base InterRepository that copies revisions from a Git into a non-Git
    repository."""

    def _target_has_shas(self, shas):
        revids = {}
        for sha in shas:
            try:
                revid = self.source.lookup_foreign_revision_id(sha)
            except NotCommitError:
                continue
            else:
                revids[revid] = sha
        return {revids[r] for r in self.target.has_revisions(revids)}

    def determine_wants_all(self, refs):
        potential = set()
        for k, v in refs.items():
            if v == ZERO_SHA:
                continue
            potential.add(self.source.controldir.get_peeled(k) or v)
        return list(potential - self._target_has_shas(potential))

    def _warn_slow(self):
        if not config.GlobalConfig().suppress_warning('slow_intervcs_push'):
            trace.warning('Fetching from Git to Bazaar repository. For better performance, fetch into a Git repository.')

    def fetch_objects(self, determine_wants, mapping, limit=None, lossy=False):
        """Fetch objects from a remote server.

        :param determine_wants: determine_wants callback
        :param mapping: BzrGitMapping to use
        :param limit: Maximum number of commits to import.
        :return: Tuple with pack hint, last imported revision id and remote
            refs
        """
        raise NotImplementedError(self.fetch_objects)

    def get_determine_wants_revids(self, revids, include_tags=False, tag_selector=None):
        wants = set()
        for revid in set(revids):
            if self.target.has_revision(revid):
                continue
            git_sha, mapping = self.source.lookup_bzr_revision_id(revid)
            wants.add(git_sha)
        return self.get_determine_wants_heads(wants, include_tags=include_tags, tag_selector=tag_selector)

    def fetch(self, revision_id=None, find_ghosts=False, mapping=None, fetch_spec=None, include_tags=False, lossy=False):
        if mapping is None:
            mapping = self.source.get_mapping()
        if revision_id is not None:
            interesting_heads = [revision_id]
        elif fetch_spec is not None:
            recipe = fetch_spec.get_recipe()
            if recipe[0] in ('search', 'proxy-search'):
                interesting_heads = recipe[1]
            else:
                raise AssertionError('Unsupported search result type %s' % recipe[0])
        else:
            interesting_heads = None
        if interesting_heads is not None:
            determine_wants = self.get_determine_wants_revids(interesting_heads, include_tags=include_tags)
        else:
            determine_wants = self.determine_wants_all
        pack_hint, _, remote_refs = self.fetch_objects(determine_wants, mapping, lossy=lossy)
        if pack_hint is not None and self.target._format.pack_compresses:
            self.target.pack(hint=pack_hint)
        result = FetchResult()
        result.refs = remote_refs
        return result