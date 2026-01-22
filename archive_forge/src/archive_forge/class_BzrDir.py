import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
class BzrDir(controldir.ControlDir):
    """A .bzr control diretory.

    BzrDir instances let you create or open any of the things that can be
    found within .bzr - checkouts, branches and repositories.

    :ivar transport:
        the transport which this bzr dir is rooted at (i.e. file:///.../.bzr/)
    :ivar root_transport:
        a transport connected to the directory this bzr was opened from
        (i.e. the parent directory holding the .bzr directory).

    Everything in the bzrdir should have the same file permissions.

    :cvar hooks: An instance of BzrDirHooks.
    """

    def break_lock(self):
        """Invoke break_lock on the first object in the bzrdir.

        If there is a tree, the tree is opened and break_lock() called.
        Otherwise, branch is tried, and finally repository.
        """
        try:
            thing_to_unlock = self.open_workingtree()
        except (errors.NotLocalUrl, errors.NoWorkingTree):
            try:
                thing_to_unlock = self.open_branch()
            except errors.NotBranchError:
                try:
                    thing_to_unlock = self.open_repository()
                except errors.NoRepositoryPresent:
                    return
        thing_to_unlock.break_lock()

    def check_conversion_target(self, target_format):
        """Check that a bzrdir as a whole can be converted to a new format."""
        target_repo_format = target_format.repository_format
        try:
            self.open_repository()._format.check_conversion_target(target_repo_format)
        except errors.NoRepositoryPresent:
            pass

    def clone_on_transport(self, transport, revision_id=None, force_new_repo=False, preserve_stacking=False, stacked_on=None, create_prefix=False, use_existing_dir=True, no_tree=False, tag_selector=None):
        """Clone this bzrdir and its contents to transport verbatim.

        :param transport: The transport for the location to produce the clone
            at.  If the target directory does not exist, it will be created.
        :param revision_id: The tip revision-id to use for any branch or
            working tree.  If not None, then the clone operation may tune
            itself to download less data.
        :param force_new_repo: Do not use a shared repository for the target,
                               even if one is available.
        :param preserve_stacking: When cloning a stacked branch, stack the
            new branch on top of the other branch's stacked-on branch.
        :param create_prefix: Create any missing directories leading up to
            to_transport.
        :param use_existing_dir: Use an existing directory if one exists.
        :param no_tree: If set to true prevents creation of a working tree.
        """
        require_stacking = stacked_on is not None
        format = self.cloning_metadir(require_stacking)
        try:
            local_repo = self.find_repository()
        except errors.NoRepositoryPresent:
            local_repo = None
        local_branches = self.get_branches()
        try:
            local_active_branch = local_branches['']
        except KeyError:
            pass
        else:
            if local_active_branch.repository.has_same_location(local_repo):
                local_repo = local_active_branch.repository
            if preserve_stacking:
                try:
                    stacked_on = local_active_branch.get_stacked_on_url()
                except (_mod_branch.UnstackableBranchFormat, errors.UnstackableRepositoryFormat, errors.NotStacked):
                    pass
        if local_repo:
            make_working_trees = local_repo.make_working_trees() and (not no_tree)
            want_shared = local_repo.is_shared()
            repo_format_name = format.repository_format.network_name()
        else:
            make_working_trees = False
            want_shared = False
            repo_format_name = None
        result_repo, result, require_stacking, repository_policy = format.initialize_on_transport_ex(transport, use_existing_dir=use_existing_dir, create_prefix=create_prefix, force_new_repo=force_new_repo, stacked_on=stacked_on, stack_on_pwd=self.root_transport.base, repo_format_name=repo_format_name, make_working_trees=make_working_trees, shared_repo=want_shared)
        if repo_format_name:
            try:
                if result_repo.user_url == result.user_url and (not require_stacking) and (revision_id is not None):
                    fetch_spec = vf_search.PendingAncestryResult([revision_id], local_repo)
                    result_repo.fetch(local_repo, fetch_spec=fetch_spec)
                else:
                    result_repo.fetch(local_repo, revision_id=revision_id)
            finally:
                result_repo.unlock()
        elif result_repo is not None:
            raise AssertionError('result_repo not None(%r)' % result_repo)
        for name, local_branch in local_branches.items():
            local_branch.clone(result, revision_id=None if name != '' else revision_id, repository_policy=repository_policy, name=name, tag_selector=tag_selector)
        try:
            result.root_transport.local_abspath('.')
            if result_repo is None or result_repo.make_working_trees():
                self.open_workingtree().clone(result, revision_id=revision_id)
        except (errors.NoWorkingTree, errors.NotLocalUrl):
            pass
        return result

    def _make_tail(self, url):
        t = _mod_transport.get_transport(url)
        t.ensure_base()

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

        def repository_policy(found_bzrdir):
            stack_on = None
            stack_on_pwd = None
            config = found_bzrdir.get_config()
            stop = False
            stack_on = config.get_default_stack_on()
            if stack_on is not None:
                stack_on_pwd = found_bzrdir.user_url
                stop = True
            try:
                repository = found_bzrdir.open_repository()
            except errors.NoRepositoryPresent:
                repository = None
            else:
                if found_bzrdir.user_url != self.user_url and (not repository.is_shared()):
                    repository = None
                    stop = True
                else:
                    stop = True
            if not stop:
                return (None, False)
            if repository:
                return (UseExistingRepository(repository, stack_on, stack_on_pwd, require_stacking=require_stacking), True)
            else:
                return (CreateRepository(self, stack_on, stack_on_pwd, require_stacking=require_stacking), True)
        if not force_new_repo:
            if stack_on is None:
                policy = self._find_containing(repository_policy)
                if policy is not None:
                    return policy
            else:
                try:
                    return UseExistingRepository(self.open_repository(), stack_on, stack_on_pwd, require_stacking=require_stacking)
                except errors.NoRepositoryPresent:
                    pass
        return CreateRepository(self, stack_on, stack_on_pwd, require_stacking=require_stacking)

    def _find_or_create_repository(self, force_new_repo):
        """Create a new repository if needed, returning the repository."""
        policy = self.determine_repository_policy(force_new_repo)
        return policy.acquire_repository()[0]

    def _find_source_repo(self, exit_stack, source_branch):
        """Find the source branch and repo for a sprout operation.

        This is helper intended for use by _sprout.

        :returns: (source_branch, source_repository).  Either or both may be
            None.  If not None, they will be read-locked (and their unlock(s)
            scheduled via the exit_stack param).
        """
        if source_branch is not None:
            exit_stack.enter_context(source_branch.lock_read())
            return (source_branch, source_branch.repository)
        try:
            source_branch = self.open_branch()
            source_repository = source_branch.repository
        except errors.NotBranchError:
            source_branch = None
            try:
                source_repository = self.open_repository()
            except errors.NoRepositoryPresent:
                source_repository = None
            else:
                exit_stack.enter_context(source_repository.lock_read())
        else:
            exit_stack.enter_context(source_branch.lock_read())
        return (source_branch, source_repository)

    def sprout(self, url, revision_id=None, force_new_repo=False, recurse='down', possible_transports=None, accelerator_tree=None, hardlink=False, stacked=False, source_branch=None, create_tree_if_local=True, lossy=False):
        """Create a copy of this controldir prepared for use as a new line of
        development.

        If url's last component does not exist, it will be created.

        Attributes related to the identity of the source branch like
        branch nickname will be cleaned, a working tree is created
        whether one existed before or not; and a local branch is always
        created.

        if revision_id is not None, then the clone operation may tune
            itself to download less data.

        :param accelerator_tree: A tree which can be used for retrieving file
            contents more quickly than the revision tree, i.e. a workingtree.
            The revision tree will be used for cases where accelerator_tree's
            content is different.
        :param hardlink: If true, hard-link files from accelerator_tree,
            where possible.
        :param stacked: If true, create a stacked branch referring to the
            location of this control directory.
        :param create_tree_if_local: If true, a working-tree will be created
            when working locally.
        :return: The created control directory
        """
        with contextlib.ExitStack() as stack:
            fetch_spec_factory = fetch.FetchSpecFactory()
            if revision_id is not None:
                fetch_spec_factory.add_revision_ids([revision_id])
                fetch_spec_factory.source_branch_stop_revision_id = revision_id
            if possible_transports is None:
                possible_transports = []
            else:
                possible_transports = list(possible_transports) + [self.root_transport]
            target_transport = _mod_transport.get_transport(url, possible_transports)
            target_transport.ensure_base()
            cloning_format = self.cloning_metadir(stacked)
            try:
                result = controldir.ControlDir.open_from_transport(target_transport)
            except errors.NotBranchError:
                result = cloning_format.initialize_on_transport(target_transport)
            source_branch, source_repository = self._find_source_repo(stack, source_branch)
            fetch_spec_factory.source_branch = source_branch
            if stacked and source_branch is not None:
                stacked_branch_url = self.root_transport.base
            else:
                stacked_branch_url = None
            repository_policy = result.determine_repository_policy(force_new_repo, stacked_branch_url, require_stacking=stacked)
            result_repo, is_new_repo = repository_policy.acquire_repository(possible_transports=possible_transports)
            stack.enter_context(result_repo.lock_write())
            fetch_spec_factory.source_repo = source_repository
            fetch_spec_factory.target_repo = result_repo
            if stacked or len(result_repo._fallback_repositories) != 0:
                target_repo_kind = fetch.TargetRepoKinds.STACKED
            elif is_new_repo:
                target_repo_kind = fetch.TargetRepoKinds.EMPTY
            else:
                target_repo_kind = fetch.TargetRepoKinds.PREEXISTING
            fetch_spec_factory.target_repo_kind = target_repo_kind
            if source_repository is not None:
                fetch_spec = fetch_spec_factory.make_fetch_spec()
                result_repo.fetch(source_repository, fetch_spec=fetch_spec)
            if source_branch is None:
                result_branch = result.create_branch()
                if revision_id is not None:
                    result_branch.generate_revision_history(revision_id)
            else:
                result_branch = source_branch.sprout(result, revision_id=revision_id, repository_policy=repository_policy, repository=result_repo)
            mutter('created new branch {!r}'.format(result_branch))
            if create_tree_if_local and (not result.has_workingtree()) and isinstance(target_transport, local.LocalTransport) and (result_repo is None or result_repo.make_working_trees()) and (result.open_branch(name='', possible_transports=possible_transports).name == result_branch.name):
                wt = result.create_workingtree(accelerator_tree=accelerator_tree, hardlink=hardlink, from_branch=result_branch)
                with wt.lock_write():
                    if not wt.is_versioned(''):
                        try:
                            wt.set_root_id(self.open_workingtree.path2id(''))
                        except errors.NoWorkingTree:
                            pass
            else:
                wt = None
            if recurse == 'down':
                tree = None
                if wt is not None:
                    tree = wt
                    basis = tree.basis_tree()
                    stack.enter_context(basis.lock_read())
                elif result_branch is not None:
                    basis = tree = result_branch.basis_tree()
                elif source_branch is not None:
                    basis = tree = source_branch.basis_tree()
                if tree is not None:
                    stack.enter_context(tree.lock_read())
                    subtrees = tree.iter_references()
                else:
                    subtrees = []
                for path in subtrees:
                    target = urlutils.join(url, urlutils.escape(path))
                    sublocation = tree.reference_parent(path, branch=result_branch, possible_transports=possible_transports)
                    if sublocation is None:
                        warning('Ignoring nested tree %s, parent location unknown.', path)
                        continue
                    sublocation.controldir.sprout(target, basis.get_reference_revision(path), force_new_repo=force_new_repo, recurse=recurse, stacked=stacked)
            return result

    def _available_backup_name(self, base):
        """Find a non-existing backup file name based on base.

        See breezy.osutils.available_backup_name about race conditions.
        """
        return osutils.available_backup_name(base, self.root_transport.has)

    def backup_bzrdir(self):
        """Backup this bzr control directory.

        :return: Tuple with old path name and new path name
        """
        with ui.ui_factory.nested_progress_bar():
            old_path = self.root_transport.abspath('.bzr')
            backup_dir = self._available_backup_name('backup.bzr')
            new_path = self.root_transport.abspath(backup_dir)
            ui.ui_factory.note(gettext('making backup of {0}\n  to {1}').format(urlutils.unescape_for_display(old_path, 'utf-8'), urlutils.unescape_for_display(new_path, 'utf-8')))
            self.root_transport.copy_tree('.bzr', backup_dir)
            return (old_path, new_path)

    def retire_bzrdir(self, limit=10000):
        """Permanently disable the bzrdir.

        This is done by renaming it to give the user some ability to recover
        if there was a problem.

        This will have horrible consequences if anyone has anything locked or
        in use.
        :param limit: number of times to retry
        """
        i = 0
        while True:
            try:
                to_path = '.bzr.retired.%d' % i
                self.root_transport.rename('.bzr', to_path)
                note(gettext('renamed {0} to {1}').format(self.root_transport.abspath('.bzr'), to_path))
                return
            except (errors.TransportError, OSError, errors.PathError):
                i += 1
                if i > limit:
                    raise
                else:
                    pass

    def _find_containing(self, evaluate):
        """Find something in a containing control directory.

        This method will scan containing control dirs, until it finds what
        it is looking for, decides that it will never find it, or runs out
        of containing control directories to check.

        It is used to implement find_repository and
        determine_repository_policy.

        :param evaluate: A function returning (value, stop).  If stop is True,
            the value will be returned.
        """
        found_bzrdir = self
        while True:
            result, stop = evaluate(found_bzrdir)
            if stop:
                return result
            next_transport = found_bzrdir.root_transport.clone('..')
            if found_bzrdir.user_url == next_transport.base:
                return None
            try:
                found_bzrdir = self.open_containing_from_transport(next_transport)[0]
            except errors.NotBranchError:
                return None

    def find_repository(self):
        """Find the repository that should be used.

        This does not require a branch as we use it to find the repo for
        new branches as well as to hook existing branches up to their
        repository.
        """

        def usable_repository(found_bzrdir):
            try:
                repository = found_bzrdir.open_repository()
            except errors.NoRepositoryPresent:
                return (None, False)
            if found_bzrdir.user_url == self.user_url:
                return (repository, True)
            elif repository.is_shared():
                return (repository, True)
            else:
                return (None, True)
        found_repo = self._find_containing(usable_repository)
        if found_repo is None:
            raise errors.NoRepositoryPresent(self)
        return found_repo

    def _find_creation_modes(self):
        """Determine the appropriate modes for files and directories.

        They're always set to be consistent with the base directory,
        assuming that this transport allows setting modes.
        """
        if self._mode_check_done:
            return
        self._mode_check_done = True
        try:
            st = self.transport.stat('.')
        except errors.TransportNotPossible:
            self._dir_mode = None
            self._file_mode = None
        else:
            if st.st_mode & 4095 == 0:
                self._dir_mode = None
                self._file_mode = None
            else:
                self._dir_mode = st.st_mode & 4095 | 448
                self._file_mode = self._dir_mode & ~3657

    def _get_file_mode(self):
        """Return Unix mode for newly created files, or None.
        """
        if not self._mode_check_done:
            self._find_creation_modes()
        return self._file_mode

    def _get_dir_mode(self):
        """Return Unix mode for newly created directories, or None.
        """
        if not self._mode_check_done:
            self._find_creation_modes()
        return self._dir_mode

    def get_config(self):
        """Get configuration for this BzrDir."""
        return config.BzrDirConfig(self)

    def _get_config(self):
        """By default, no configuration is available."""
        return None

    def __init__(self, _transport, _format):
        """Initialize a Bzr control dir object.

        Only really common logic should reside here, concrete classes should be
        made with varying behaviours.

        :param _format: the format that is creating this BzrDir instance.
        :param _transport: the transport this dir is based at.
        """
        self._format = _format
        self.transport = _transport.clone('.bzr')
        self.root_transport = _transport
        self._mode_check_done = False

    @property
    def user_transport(self):
        return self.root_transport

    @property
    def control_transport(self):
        return self.transport

    def _cloning_metadir(self):
        """Produce a metadir suitable for cloning with.

        :returns: (destination_bzrdir_format, source_repository)
        """
        result_format = self._format.__class__()
        try:
            try:
                branch = self.open_branch(ignore_fallbacks=True)
                source_repository = branch.repository
                result_format._branch_format = branch._format
            except errors.NotBranchError:
                source_repository = self.open_repository()
        except errors.NoRepositoryPresent:
            source_repository = None
        else:
            repo_format = source_repository._format
            if isinstance(repo_format, remote.RemoteRepositoryFormat):
                source_repository._ensure_real()
                repo_format = source_repository._real_repository._format
            result_format.repository_format = repo_format
        try:
            tree = self.open_workingtree(recommend_upgrade=False)
        except (errors.NoWorkingTree, errors.NotLocalUrl):
            result_format.workingtree_format = None
        else:
            result_format.workingtree_format = tree._format.__class__()
        return (result_format, source_repository)

    def cloning_metadir(self, require_stacking=False):
        """Produce a metadir suitable for cloning or sprouting with.

        These operations may produce workingtrees (yes, even though they're
        "cloning" something that doesn't have a tree), so a viable workingtree
        format must be selected.

        :require_stacking: If True, non-stackable formats will be upgraded
            to similar stackable formats.
        :returns: a ControlDirFormat with all component formats either set
            appropriately or set to None if that component should not be
            created.
        """
        format, repository = self._cloning_metadir()
        if format._workingtree_format is None:
            if repository is None:
                return format
            tree_format = repository._format._matchingcontroldir.workingtree_format
            format.workingtree_format = tree_format.__class__()
        if require_stacking:
            format.require_stacking()
        return format

    def get_branch_transport(self, branch_format, name=None):
        """Get the transport for use by branch format in this BzrDir.

        Note that bzr dirs that do not support format strings will raise
        IncompatibleFormat if the branch format they are given has
        a format string, and vice versa.

        If branch_format is None, the transport is returned with no
        checking. If it is not None, then the returned transport is
        guaranteed to point to an existing directory ready for use.
        """
        raise NotImplementedError(self.get_branch_transport)

    def get_repository_transport(self, repository_format):
        """Get the transport for use by repository format in this BzrDir.

        Note that bzr dirs that do not support format strings will raise
        IncompatibleFormat if the repository format they are given has
        a format string, and vice versa.

        If repository_format is None, the transport is returned with no
        checking. If it is not None, then the returned transport is
        guaranteed to point to an existing directory ready for use.
        """
        raise NotImplementedError(self.get_repository_transport)

    def get_workingtree_transport(self, tree_format):
        """Get the transport for use by workingtree format in this BzrDir.

        Note that bzr dirs that do not support format strings will raise
        IncompatibleFormat if the workingtree format they are given has a
        format string, and vice versa.

        If workingtree_format is None, the transport is returned with no
        checking. If it is not None, then the returned transport is
        guaranteed to point to an existing directory ready for use.
        """
        raise NotImplementedError(self.get_workingtree_transport)

    @classmethod
    def create(cls, base, format=None, possible_transports=None) -> 'BzrDir':
        """Create a new BzrDir at the url 'base'.

        :param format: If supplied, the format of branch to create.  If not
            supplied, the default is used.
        :param possible_transports: If supplied, a list of transports that
            can be reused to share a remote connection.
        """
        if cls is not BzrDir:
            raise AssertionError('BzrDir.create always creates the default format, not one of %r' % cls)
        if format is None:
            format = BzrDirFormat.get_default_format()
        return cast('BzrDir', controldir.ControlDir.create(base, format=format, possible_transports=possible_transports))

    def __repr__(self):
        return '<{} at {!r}>'.format(self.__class__.__name__, self.user_url)

    def update_feature_flags(self, updated_flags):
        """Update the features required by this bzrdir.

        :param updated_flags: Dictionary mapping feature names to necessities
            A necessity can be None to indicate the feature should be removed
        """
        self.control_files.lock_write()
        try:
            self._format._update_feature_flags(updated_flags)
            self.transport.put_bytes('branch-format', self._format.as_string())
        finally:
            self.control_files.unlock()