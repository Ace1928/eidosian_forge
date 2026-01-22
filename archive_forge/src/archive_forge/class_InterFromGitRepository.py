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
class InterFromGitRepository(InterRepository):
    _matching_repo_format = GitRepositoryFormat()

    def _target_has_shas(self, shas):
        raise NotImplementedError(self._target_has_shas)

    def get_determine_wants_heads(self, wants, include_tags=False, tag_selector=None):
        wants = set(wants)

        def determine_wants(refs):
            unpeel_lookup = {}
            for k, v in refs.items():
                if k.endswith(PEELED_TAG_SUFFIX):
                    unpeel_lookup[v] = refs[k[:-len(PEELED_TAG_SUFFIX)]]
            potential = {unpeel_lookup.get(w, w) for w in wants}
            if include_tags:
                for k, sha in refs.items():
                    if k.endswith(PEELED_TAG_SUFFIX):
                        continue
                    try:
                        tag_name = ref_to_tag_name(k)
                    except ValueError:
                        continue
                    if tag_selector and (not tag_selector(tag_name)):
                        continue
                    if sha == ZERO_SHA:
                        continue
                    potential.add(sha)
            return list(potential - self._target_has_shas(potential))
        return determine_wants

    def determine_wants_all(self, refs):
        raise NotImplementedError(self.determine_wants_all)

    @staticmethod
    def _get_repo_format_to_test():
        return None

    def copy_content(self, revision_id=None):
        """See InterRepository.copy_content."""
        self.fetch(revision_id, find_ghosts=False)

    def search_missing_revision_ids(self, find_ghosts=True, revision_ids=None, if_present_ids=None, limit=None):
        if limit is not None:
            raise FetchLimitUnsupported(self)
        if revision_ids is None and if_present_ids is None:
            todo = set(self.source.all_revision_ids())
        else:
            todo = set()
            if revision_ids is not None:
                for revid in revision_ids:
                    if not self.source.has_revision(revid):
                        raise NoSuchRevision(revid, self.source)
                todo.update(revision_ids)
            if if_present_ids is not None:
                todo.update(if_present_ids)
        result_set = todo.difference(self.target.all_revision_ids())
        result_parents = set(itertools.chain.from_iterable(self.source.get_graph().get_parent_map(result_set).values()))
        included_keys = result_set.intersection(result_parents)
        start_keys = result_set.difference(included_keys)
        exclude_keys = result_parents.difference(result_set)
        return GitSearchResult(start_keys, exclude_keys, result_set)