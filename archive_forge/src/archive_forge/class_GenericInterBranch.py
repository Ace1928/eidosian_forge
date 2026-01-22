from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
class GenericInterBranch(InterBranch):
    """InterBranch implementation that uses public Branch functions."""

    @classmethod
    def is_compatible(klass, source, target):
        return True

    @classmethod
    def _get_branch_formats_to_test(klass):
        return [(format_registry.get_default(), format_registry.get_default())]

    @classmethod
    def unwrap_format(klass, format):
        if isinstance(format, remote.RemoteBranchFormat):
            format._ensure_real()
            return format._custom_format
        return format

    def copy_content_into(self, revision_id=None, tag_selector=None):
        """Copy the content of source into target

        revision_id: if not None, the revision history in the new branch will
                     be truncated to end with revision_id.
        """
        with self.source.lock_read(), self.target.lock_write():
            self.source._synchronize_history(self.target, revision_id)
            self.update_references()
            try:
                parent = self.source.get_parent()
            except errors.InaccessibleParent as e:
                mutter('parent was not accessible to copy: %s', str(e))
            else:
                if parent:
                    self.target.set_parent(parent)
            if self.source._push_should_merge_tags():
                self.source.tags.merge_to(self.target.tags, selector=tag_selector)

    def fetch(self, stop_revision=None, limit=None, lossy=False):
        if self.target.base == self.source.base:
            return (0, [])
        with self.source.lock_read(), self.target.lock_write():
            fetch_spec_factory = fetch.FetchSpecFactory()
            fetch_spec_factory.source_branch = self.source
            fetch_spec_factory.source_branch_stop_revision_id = stop_revision
            fetch_spec_factory.source_repo = self.source.repository
            fetch_spec_factory.target_repo = self.target.repository
            fetch_spec_factory.target_repo_kind = fetch.TargetRepoKinds.PREEXISTING
            fetch_spec_factory.limit = limit
            fetch_spec = fetch_spec_factory.make_fetch_spec()
            return self.target.repository.fetch(self.source.repository, lossy=lossy, fetch_spec=fetch_spec)

    def _update_revisions(self, stop_revision=None, overwrite=False, graph=None):
        with self.source.lock_read(), self.target.lock_write():
            other_revno, other_last_revision = self.source.last_revision_info()
            stop_revno = None
            if stop_revision is None:
                stop_revision = other_last_revision
                if _mod_revision.is_null(stop_revision):
                    return
                stop_revno = other_revno
            last_rev = self.target.last_revision()
            self.fetch(stop_revision=stop_revision)
            if not overwrite:
                if graph is None:
                    graph = self.target.repository.get_graph()
                if self.target._check_if_descendant_or_diverged(stop_revision, last_rev, graph, self.source):
                    return
            if stop_revno is None:
                if graph is None:
                    graph = self.target.repository.get_graph()
                this_revno, this_last_revision = self.target.last_revision_info()
                stop_revno = graph.find_distance_to_null(stop_revision, [(other_last_revision, other_revno), (this_last_revision, this_revno)])
            self.target.set_last_revision_info(stop_revno, stop_revision)

    def pull(self, overwrite=False, stop_revision=None, possible_transports=None, run_hooks=True, _override_hook_target=None, local=False, tag_selector=None):
        """Pull from source into self, updating my master if any.

        Args:
          run_hooks: Private parameter - if false, this branch
            is being called because it's the master of the primary branch,
            so it should not run its hooks.
        """
        with contextlib.ExitStack() as exit_stack:
            exit_stack.enter_context(self.target.lock_write())
            bound_location = self.target.get_bound_location()
            if local and (not bound_location):
                raise errors.LocalRequiresBoundBranch()
            master_branch = None
            source_is_master = False
            if bound_location:
                normalized = urlutils.normalize_url(bound_location)
                try:
                    relpath = self.source.user_transport.relpath(normalized)
                    source_is_master = relpath == ''
                except (errors.PathNotChild, urlutils.InvalidURL):
                    source_is_master = False
            if not local and bound_location and (not source_is_master):
                master_branch = self.target.get_master_branch(possible_transports)
                exit_stack.enter_context(master_branch.lock_write())
            if master_branch:
                master_branch.pull(self.source, overwrite=overwrite, stop_revision=stop_revision, run_hooks=False, tag_selector=tag_selector)
            return self._pull(overwrite, stop_revision, _hook_master=master_branch, run_hooks=run_hooks, _override_hook_target=_override_hook_target, merge_tags_to_master=not source_is_master, tag_selector=tag_selector)

    def push(self, overwrite=False, stop_revision=None, lossy=False, _override_hook_source_branch=None, tag_selector=None):
        """See InterBranch.push.

        This is the basic concrete implementation of push()

        Args:
          _override_hook_source_branch: If specified, run the hooks
            passing this Branch as the source, rather than self.  This is for
            use of RemoteBranch, where push is delegated to the underlying
            vfs-based Branch.
        """
        if lossy:
            raise errors.LossyPushToSameVCS(self.source, self.target)

        def _run_hooks():
            if _override_hook_source_branch:
                result.source_branch = _override_hook_source_branch
            for hook in Branch.hooks['post_push']:
                hook(result)
        with self.source.lock_read(), self.target.lock_write():
            bound_location = self.target.get_bound_location()
            if bound_location and self.target.base != bound_location:
                master_branch = self.target.get_master_branch()
                with master_branch.lock_write():
                    master_inter = InterBranch.get(self.source, master_branch)
                    master_inter._basic_push(overwrite, stop_revision, tag_selector=tag_selector)
                    result = self._basic_push(overwrite, stop_revision, tag_selector=tag_selector)
                    result.master_branch = master_branch
                    result.local_branch = self.target
                    _run_hooks()
            else:
                master_branch = None
                result = self._basic_push(overwrite, stop_revision, tag_selector=tag_selector)
                result.master_branch = self.target
                result.local_branch = None
                _run_hooks()
            return result

    def _basic_push(self, overwrite, stop_revision, tag_selector=None):
        """Basic implementation of push without bound branches or hooks.

        Must be called with source read locked and target write locked.
        """
        result = BranchPushResult()
        result.source_branch = self.source
        result.target_branch = self.target
        result.old_revno, result.old_revid = self.target.last_revision_info()
        overwrite = _fix_overwrite_type(overwrite)
        if result.old_revid != stop_revision:
            graph = self.source.repository.get_graph(self.target.repository)
            self._update_revisions(stop_revision, overwrite='history' in overwrite, graph=graph)
        if self.source._push_should_merge_tags():
            result.tag_updates, result.tag_conflicts = self.source.tags.merge_to(self.target.tags, 'tags' in overwrite, selector=tag_selector)
        self.update_references()
        result.new_revno, result.new_revid = self.target.last_revision_info()
        return result

    def _pull(self, overwrite=False, stop_revision=None, possible_transports=None, _hook_master=None, run_hooks=True, _override_hook_target=None, local=False, merge_tags_to_master=True, tag_selector=None):
        """See Branch.pull.

        This function is the core worker, used by GenericInterBranch.pull to
        avoid duplication when pulling source->master and source->local.

        Args:
          _hook_master: Private parameter - set the branch to
            be supplied as the master to pull hooks.
          run_hooks: Private parameter - if false, this branch
            is being called because it's the master of the primary branch,
            so it should not run its hooks.
            is being called because it's the master of the primary branch,
            so it should not run its hooks.
          _override_hook_target: Private parameter - set the branch to be
            supplied as the target_branch to pull hooks.
          local: Only update the local branch, and not the bound branch.
        """
        if local:
            raise errors.LocalRequiresBoundBranch()
        result = PullResult()
        result.source_branch = self.source
        if _override_hook_target is None:
            result.target_branch = self.target
        else:
            result.target_branch = _override_hook_target
        with self.source.lock_read():
            graph = self.target.repository.get_graph(self.source.repository)
            result.old_revno, result.old_revid = self.target.last_revision_info()
            overwrite = _fix_overwrite_type(overwrite)
            self._update_revisions(stop_revision, overwrite='history' in overwrite, graph=graph)
            result.tag_updates, result.tag_conflicts = self.source.tags.merge_to(self.target.tags, 'tags' in overwrite, ignore_master=not merge_tags_to_master, selector=tag_selector)
            self.update_references()
            result.new_revno, result.new_revid = self.target.last_revision_info()
            if _hook_master:
                result.master_branch = _hook_master
                result.local_branch = result.target_branch
            else:
                result.master_branch = result.target_branch
                result.local_branch = None
            if run_hooks:
                for hook in Branch.hooks['post_pull']:
                    hook(result)
            return result

    def update_references(self):
        if not getattr(self.source._format, 'supports_reference_locations', False):
            return
        reference_dict = self.source._get_all_reference_info()
        if len(reference_dict) == 0:
            return
        old_base = self.source.base
        new_base = self.target.base
        target_reference_dict = self.target._get_all_reference_info()
        for tree_path, (branch_location, file_id) in reference_dict.items():
            try:
                branch_location = urlutils.rebase_url(branch_location, old_base, new_base)
            except urlutils.InvalidRebaseURLs:
                branch_location = urlutils.join(old_base, branch_location)
            target_reference_dict.setdefault(tree_path, (branch_location, file_id))
        self.target._set_all_reference_info(target_reference_dict)