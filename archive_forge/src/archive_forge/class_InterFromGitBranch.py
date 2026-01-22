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
class InterFromGitBranch(branch.GenericInterBranch):
    """InterBranch implementation that pulls from Git into bzr."""

    @staticmethod
    def _get_branch_formats_to_test():
        try:
            default_format = branch.format_registry.get_default()
        except AttributeError:
            default_format = branch.BranchFormat._default_format
        from .remote import RemoteGitBranchFormat
        return [(RemoteGitBranchFormat(), default_format), (LocalGitBranchFormat(), default_format)]

    @classmethod
    def _get_interrepo(self, source, target):
        return _mod_repository.InterRepository.get(source.repository, target.repository)

    @classmethod
    def is_compatible(cls, source, target):
        if not isinstance(source, GitBranch):
            return False
        if isinstance(target, GitBranch):
            return False
        if getattr(cls._get_interrepo(source, target), 'fetch_objects', None) is None:
            return False
        return True

    def fetch(self, stop_revision=None, fetch_tags=None, limit=None, lossy=False):
        self.fetch_objects(stop_revision, fetch_tags=fetch_tags, limit=limit, lossy=lossy)
        return _mod_repository.FetchResult()

    def fetch_objects(self, stop_revision, fetch_tags, limit=None, lossy=False, tag_selector=None):
        interrepo = self._get_interrepo(self.source, self.target)
        if fetch_tags is None:
            c = self.source.get_config_stack()
            fetch_tags = c.get('branch.fetch_tags')

        def determine_wants(heads):
            if stop_revision is None:
                try:
                    head = heads[self.source.ref]
                except KeyError:
                    self._last_revid = revision.NULL_REVISION
                else:
                    self._last_revid = self.source.lookup_foreign_revision_id(head)
            else:
                self._last_revid = stop_revision
            real = interrepo.get_determine_wants_revids([self._last_revid], include_tags=fetch_tags, tag_selector=tag_selector)
            return real(heads)
        pack_hint, head, refs = interrepo.fetch_objects(determine_wants, self.source.mapping, limit=limit, lossy=lossy)
        if pack_hint is not None and self.target.repository._format.pack_compresses:
            self.target.repository.pack(hint=pack_hint)
        return (head, refs)

    def _update_revisions(self, stop_revision=None, overwrite=False, tag_selector=None):
        head, refs = self.fetch_objects(stop_revision, fetch_tags=None, tag_selector=tag_selector)
        _update_tip(self.source, self.target, self._last_revid, overwrite)
        return (head, refs)

    def update_references(self, revid=None):
        if revid is None:
            revid = self.target.last_revision()
        tree = self.target.repository.revision_tree(revid)
        try:
            with tree.get_file('.gitmodules') as f:
                for path, url, section in parse_submodules(GitConfigFile.from_file(f)):
                    self.target.set_reference_info(tree.path2id(decode_git_path(path)), url.decode('utf-8'), decode_git_path(path))
        except transport.NoSuchFile:
            pass

    def _basic_pull(self, stop_revision, overwrite, run_hooks, _override_hook_target, _hook_master, tag_selector=None):
        if overwrite is True:
            overwrite = {'history', 'tags'}
        elif not overwrite:
            overwrite = set()
        result = GitBranchPullResult()
        result.source_branch = self.source
        if _override_hook_target is None:
            result.target_branch = self.target
        else:
            result.target_branch = _override_hook_target
        with self.target.lock_write(), self.source.lock_read():
            result.old_revno, result.old_revid = self.target.last_revision_info()
            result.new_git_head, remote_refs = self._update_revisions(stop_revision, overwrite='history' in overwrite, tag_selector=tag_selector)
            tags_ret = self.source.tags.merge_to(self.target.tags, 'tags' in overwrite, ignore_master=True)
            if isinstance(tags_ret, tuple):
                result.tag_updates, result.tag_conflicts = tags_ret
            else:
                result.tag_conflicts = tags_ret
            result.new_revno, result.new_revid = self.target.last_revision_info()
            self.update_references(revid=result.new_revid)
            if _hook_master:
                result.master_branch = _hook_master
                result.local_branch = result.target_branch
            else:
                result.master_branch = result.target_branch
                result.local_branch = None
            if run_hooks:
                for hook in branch.Branch.hooks['post_pull']:
                    hook(result)
            return result

    def pull(self, overwrite=False, stop_revision=None, possible_transports=None, _hook_master=None, run_hooks=True, _override_hook_target=None, local=False, tag_selector=None):
        """See Branch.pull.

        :param _hook_master: Private parameter - set the branch to
            be supplied as the master to pull hooks.
        :param run_hooks: Private parameter - if false, this branch
            is being called because it's the master of the primary branch,
            so it should not run its hooks.
        :param _override_hook_target: Private parameter - set the branch to be
            supplied as the target_branch to pull hooks.
        """
        bound_location = self.target.get_bound_location()
        if local and (not bound_location):
            raise errors.LocalRequiresBoundBranch()
        source_is_master = False
        with contextlib.ExitStack() as es:
            es.enter_context(self.source.lock_read())
            if bound_location:
                normalized = urlutils.normalize_url(bound_location)
                try:
                    relpath = self.source.user_transport.relpath(normalized)
                    source_is_master = relpath == ''
                except (errors.PathNotChild, urlutils.InvalidURL):
                    source_is_master = False
            if not local and bound_location and (not source_is_master):
                master_branch = self.target.get_master_branch(possible_transports)
                es.enter_context(master_branch.lock_write())
                master_branch.pull(self.source, overwrite=overwrite, stop_revision=stop_revision, run_hooks=False)
            else:
                master_branch = None
            return self._basic_pull(stop_revision, overwrite, run_hooks, _override_hook_target, _hook_master=master_branch, tag_selector=tag_selector)

    def _basic_push(self, overwrite, stop_revision, tag_selector=None):
        if overwrite is True:
            overwrite = {'history', 'tags'}
        elif not overwrite:
            overwrite = set()
        result = branch.BranchPushResult()
        result.source_branch = self.source
        result.target_branch = self.target
        result.old_revno, result.old_revid = self.target.last_revision_info()
        result.new_git_head, remote_refs = self._update_revisions(stop_revision, overwrite='history' in overwrite, tag_selector=tag_selector)
        tags_ret = self.source.tags.merge_to(self.target.tags, 'tags' in overwrite, ignore_master=True, selector=tag_selector)
        result.tag_updates, result.tag_conflicts = tags_ret
        result.new_revno, result.new_revid = self.target.last_revision_info()
        self.update_references(revid=result.new_revid)
        return result