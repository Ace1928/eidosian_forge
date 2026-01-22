import contextlib
import os
from dulwich.refs import SymrefLoop
from .. import branch as _mod_branch
from .. import errors as brz_errors
from .. import osutils, trace, urlutils
from ..controldir import (BranchReferenceLoop, ControlDir, ControlDirFormat,
from ..transport import (FileExists, NoSuchFile, do_catching_redirections,
from .mapping import decode_git_path, encode_git_path
from .push import GitPushResult
from .transportgit import OBJECTDIR, TransportObjectStore
class GitDir(ControlDir):
    """An adapter to the '.git' dir used by git."""

    @property
    def control_transport(self):
        return self.transport

    def is_supported(self):
        return True

    def can_convert_format(self):
        return False

    def break_lock(self):
        raise NotImplementedError(self.break_lock)

    def cloning_metadir(self, stacked=False):
        return format_registry.make_controldir('git')

    def checkout_metadir(self, stacked=False):
        return format_registry.make_controldir('git')

    def _get_selected_ref(self, branch, ref=None):
        if ref is not None and branch is not None:
            raise brz_errors.BzrError("can't specify both ref and branch")
        if ref is not None:
            return ref
        if branch is not None:
            from .refs import branch_name_to_ref
            return branch_name_to_ref(branch)
        segment_parameters = getattr(self.user_transport, 'get_segment_parameters', lambda: {})()
        ref = segment_parameters.get('ref')
        if ref is not None:
            return urlutils.unquote_to_bytes(ref)
        if branch is None and getattr(self, '_get_selected_branch', False):
            branch = self._get_selected_branch()
            if branch is not None:
                from .refs import branch_name_to_ref
                return branch_name_to_ref(branch)
        return b'HEAD'

    def get_config(self):
        return GitDirConfig()

    def _available_backup_name(self, base):
        return osutils.available_backup_name(base, self.root_transport.has)

    def sprout(self, url, revision_id=None, force_new_repo=False, recurse='down', possible_transports=None, accelerator_tree=None, hardlink=False, stacked=False, source_branch=None, create_tree_if_local=True):
        from ..repository import InterRepository
        from ..transport import get_transport
        from ..transport.local import LocalTransport
        target_transport = get_transport(url, possible_transports)
        target_transport.ensure_base()
        cloning_format = self.cloning_metadir()
        try:
            result = ControlDir.open_from_transport(target_transport)
        except brz_errors.NotBranchError:
            result = cloning_format.initialize_on_transport(target_transport)
        if source_branch is None:
            source_branch = self.open_branch()
        source_repository = self.find_repository()
        try:
            result_repo = result.find_repository()
        except brz_errors.NoRepositoryPresent:
            result_repo = result.create_repository()
        if stacked:
            raise _mod_branch.UnstackableBranchFormat(self._format, self.user_url)
        interrepo = InterRepository.get(source_repository, result_repo)
        if revision_id is not None:
            determine_wants = interrepo.get_determine_wants_revids([revision_id], include_tags=True)
        else:
            determine_wants = interrepo.determine_wants_all
        interrepo.fetch_objects(determine_wants=determine_wants, mapping=source_branch.mapping)
        result_branch = source_branch.sprout(result, revision_id=revision_id, repository=result_repo)
        if create_tree_if_local and result.open_branch(name='').name == result_branch.name and isinstance(target_transport, LocalTransport) and (result_repo is None or result_repo.make_working_trees()):
            wt = result.create_workingtree(accelerator_tree=accelerator_tree, hardlink=hardlink, from_branch=result_branch)
        else:
            wt = None
        if recurse == 'down':
            with contextlib.ExitStack() as stack:
                basis = None
                if wt is not None:
                    basis = wt.basis_tree()
                elif result_branch is not None:
                    basis = result_branch.basis_tree()
                elif source_branch is not None:
                    basis = source_branch.basis_tree()
                if basis is not None:
                    stack.enter_context(basis.lock_read())
                    subtrees = basis.iter_references()
                else:
                    subtrees = []
                for path in subtrees:
                    target = urlutils.join(url, urlutils.escape(path))
                    sublocation = wt.get_reference_info(path)
                    if sublocation is None:
                        trace.warning('Unable to find submodule info for %s', path)
                        continue
                    remote_url = urlutils.join(self.user_url, sublocation)
                    try:
                        subbranch = _mod_branch.Branch.open(remote_url, possible_transports=possible_transports)
                    except brz_errors.NotBranchError as e:
                        trace.warning('Unable to clone submodule %s from %s: %s', path, remote_url, e)
                        continue
                    subbranch.controldir.sprout(target, basis.get_reference_revision(path), force_new_repo=force_new_repo, recurse=recurse, stacked=stacked)
        if getattr(result_repo, '_git', None):
            result_repo._git.object_store.close()
        return result

    def clone_on_transport(self, transport, revision_id=None, force_new_repo=False, preserve_stacking=False, stacked_on=None, create_prefix=False, use_existing_dir=True, no_tree=False, tag_selector=None):
        """See ControlDir.clone_on_transport."""
        from ..repository import InterRepository
        from ..transport.local import LocalTransport
        from .mapping import default_mapping
        from .refs import is_peeled
        if no_tree:
            format = BareLocalGitControlDirFormat()
        else:
            format = LocalGitControlDirFormat()
        if stacked_on is not None:
            raise _mod_branch.UnstackableBranchFormat(format, self.user_url)
        target_repo, target_controldir, stacking, repo_policy = format.initialize_on_transport_ex(transport, use_existing_dir=use_existing_dir, create_prefix=create_prefix, force_new_repo=force_new_repo)
        target_repo = target_controldir.find_repository()
        target_git_repo = target_repo._git
        source_repo = self.find_repository()
        interrepo = InterRepository.get(source_repo, target_repo)
        if revision_id is not None:
            determine_wants = interrepo.get_determine_wants_revids([revision_id], include_tags=True, tag_selector=tag_selector)
        else:
            determine_wants = interrepo.determine_wants_all
        pack_hint, _, refs = interrepo.fetch_objects(determine_wants, mapping=default_mapping)
        for name, val in refs.items():
            if is_peeled(name):
                continue
            if val in target_git_repo.object_store:
                target_git_repo.refs[name] = val
        result_dir = LocalGitDir(transport, target_git_repo, format)
        result_branch = result_dir.open_branch()
        try:
            parent = self.open_branch().get_parent()
        except brz_errors.InaccessibleParent:
            pass
        else:
            if parent:
                result_branch.set_parent(parent)
        if revision_id is not None:
            result_branch.set_last_revision(revision_id)
        if not no_tree and isinstance(result_dir.root_transport, LocalTransport):
            if result_dir.open_repository().make_working_trees():
                try:
                    local_wt = self.open_workingtree()
                except brz_errors.NoWorkingTree:
                    pass
                except brz_errors.NotLocalUrl:
                    result_dir.create_workingtree(revision_id=revision_id)
                else:
                    local_wt.clone(result_dir, revision_id=revision_id)
        return result_dir

    def find_repository(self):
        """Find the repository that should be used.

        This does not require a branch as we use it to find the repo for
        new branches as well as to hook existing branches up to their
        repository.
        """
        return self._gitrepository_class(self._find_commondir())

    def get_refs_container(self):
        """Retrieve the refs container.
        """
        raise NotImplementedError(self.get_refs_container)

    def determine_repository_policy(self, force_new_repo=False, stack_on=None, stack_on_pwd=None, require_stacking=False):
        """Return an object representing a policy to use.

        This controls whether a new repository is created, and the format of
        that repository, or some existing shared repository used instead.

        If stack_on is supplied, will not seek a containing shared repo.

        :param force_new_repo: If True, require a new repository to be created.
        :param stack_on: If supplied, the location to stack on.  If not
            supplied, a default_stack_on location may be used.
        :param stack_on_pwd: If stack_on is relative, the location it is
            relative to.
        """
        return UseExistingRepository(self.find_repository())

    def branch_names(self):
        from .refs import ref_to_branch_name
        ret = []
        for ref in self.get_refs_container().keys():
            try:
                branch_name = ref_to_branch_name(ref)
            except UnicodeDecodeError:
                trace.warning('Ignoring branch %r with unicode error ref', ref)
                continue
            except ValueError:
                continue
            ret.append(branch_name)
        return ret

    def get_branches(self):
        from .refs import ref_to_branch_name
        ret = {}
        for ref in self.get_refs_container().keys():
            try:
                branch_name = ref_to_branch_name(ref)
            except UnicodeDecodeError:
                trace.warning('Ignoring branch %r with unicode error ref', ref)
                continue
            except ValueError:
                continue
            ret[branch_name] = self.open_branch(ref=ref)
        return ret

    def list_branches(self):
        return list(self.get_branches().values())

    def push_branch(self, source, revision_id=None, overwrite=False, remember=False, create_prefix=False, lossy=False, name=None, tag_selector=None):
        """Push the source branch into this ControlDir."""
        push_result = GitPushResult()
        push_result.workingtree_updated = None
        push_result.master_branch = None
        push_result.source_branch = source
        push_result.stacked_on = None
        from .branch import GitBranch
        if isinstance(source, GitBranch) and lossy:
            raise brz_errors.LossyPushToSameVCS(source.controldir, self)
        target = self.open_branch(name, nascent_ok=True)
        push_result.branch_push_result = source.push(target, overwrite=overwrite, stop_revision=revision_id, lossy=lossy, tag_selector=tag_selector)
        push_result.new_revid = push_result.branch_push_result.new_revid
        push_result.old_revid = push_result.branch_push_result.old_revid
        try:
            wt = self.open_workingtree()
        except brz_errors.NoWorkingTree:
            push_result.workingtree_updated = None
        else:
            if self.open_branch(name='').name == target.name:
                wt._update_git_tree(old_revision=push_result.old_revid, new_revision=push_result.new_revid)
                push_result.workingtree_updated = True
            else:
                push_result.workingtree_updated = False
        push_result.target_branch = target
        if source.get_push_location() is None or remember:
            source.set_push_location(push_result.target_branch.base)
        return push_result