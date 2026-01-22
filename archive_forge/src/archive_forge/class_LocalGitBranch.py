import contextlib
from collections import defaultdict
from functools import partial
from io import BytesIO
from typing import Dict, Optional, Set, Tuple
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.objects import ZERO_SHA, NotCommitError
from dulwich.repo import check_ref_format
from .. import branch, config, controldir, errors, lock
from .. import repository as _mod_repository
from .. import revision, trace, transport, urlutils
from ..foreign import ForeignBranch
from ..revision import NULL_REVISION
from ..tag import InterTags, TagConflict, Tags, TagSelector, TagUpdates
from ..trace import is_quiet, mutter, warning
from .errors import NoPushSupport
from .mapping import decode_git_path, encode_git_path
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_tag, ref_to_branch_name,
from .unpeel_map import UnpeelMap
from .urls import bzr_url_to_git_url, git_url_to_bzr_url
class LocalGitBranch(GitBranch):
    """A local Git branch."""

    def __init__(self, controldir, repository, ref):
        super().__init__(controldir, repository, ref, LocalGitBranchFormat())

    def create_checkout(self, to_location, revision_id=None, lightweight=False, accelerator_tree=None, hardlink=False):
        t = transport.get_transport(to_location)
        t.ensure_base()
        format = self._get_checkout_format(lightweight=lightweight)
        checkout = format.initialize_on_transport(t)
        if lightweight:
            from_branch = checkout.set_branch_reference(target_branch=self)
        else:
            policy = checkout.determine_repository_policy()
            policy.acquire_repository()
            checkout_branch = checkout.create_branch()
            checkout_branch.bind(self)
            checkout_branch.pull(self, stop_revision=revision_id)
            from_branch = None
        return checkout.create_workingtree(revision_id, from_branch=from_branch, hardlink=hardlink)

    def _lock_ref(self):
        self._ref_lock = self.repository._git.refs.lock_ref(self.ref)

    def _unlock_ref(self):
        self._ref_lock.unlock()

    def break_lock(self):
        self.repository._git.refs.unlock_ref(self.ref)

    def _gen_revision_history(self):
        if self.head is None:
            return []
        last_revid = self.last_revision()
        graph = self.repository.get_graph()
        try:
            ret = list(graph.iter_lefthand_ancestry(last_revid, (revision.NULL_REVISION,)))
        except errors.RevisionNotPresent as e:
            raise errors.GhostRevisionsHaveNoRevno(last_revid, e.revision_id)
        ret.reverse()
        return ret

    def _get_head(self):
        try:
            return self.repository._git.refs[self.ref]
        except KeyError:
            return None

    def _read_last_revision_info(self):
        last_revid = self.last_revision()
        graph = self.repository.get_graph()
        try:
            revno = graph.find_distance_to_null(last_revid, [(revision.NULL_REVISION, 0)])
        except errors.GhostRevisionsHaveNoRevno:
            revno = None
        return (revno, last_revid)

    def set_last_revision_info(self, revno, revision_id):
        self.set_last_revision(revision_id)
        self._last_revision_info_cache = (revno, revision_id)

    def set_last_revision(self, revid):
        if not revid or not isinstance(revid, bytes):
            raise errors.InvalidRevisionId(revision_id=revid, branch=self)
        if revid == NULL_REVISION:
            newhead = None
        else:
            newhead, self.mapping = self.repository.lookup_bzr_revision_id(revid)
            if self.mapping is None:
                raise AssertionError
        self._set_head(newhead)

    def _set_head(self, value):
        if value == ZERO_SHA:
            raise ValueError(value)
        self._head = value
        if value is None:
            del self.repository._git.refs[self.ref]
        else:
            self.repository._git.refs[self.ref] = self._head
        self._clear_cached_state()
    head = property(_get_head, _set_head)

    def get_push_location(self):
        """See Branch.get_push_location."""
        push_loc = self.get_config_stack().get('push_location')
        if push_loc is not None:
            return push_loc
        cs = self.repository._git.get_config_stack()
        return self._get_related_push_branch(cs)

    def set_push_location(self, location):
        """See Branch.set_push_location."""
        self.get_config().set_user_option('push_location', location, store=config.STORE_LOCATION)

    def supports_tags(self):
        return True

    def store_uncommitted(self, creator):
        raise errors.StoringUncommittedNotSupported(self)

    def _iter_tag_refs(self):
        """Iterate over the tag refs.

        :param refs: Refs dictionary (name -> git sha1)
        :return: iterator over (ref_name, tag_name, peeled_sha1, unpeeled_sha1)
        """
        refs = self.repository.controldir.get_refs_container()
        for ref_name, unpeeled in refs.as_dict().items():
            try:
                tag_name = ref_to_tag_name(ref_name)
            except (ValueError, UnicodeDecodeError):
                continue
            peeled = refs.get_peeled(ref_name)
            if peeled is None:
                peeled = unpeeled
            if not isinstance(tag_name, str):
                raise TypeError(tag_name)
            yield (ref_name, tag_name, peeled, unpeeled)

    def create_memorytree(self):
        from .memorytree import GitMemoryTree
        return GitMemoryTree(self, self.repository._git.object_store, self.head)