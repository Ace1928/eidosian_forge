from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from .. import errors, lockable_files
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import urlutils
from ..branch import (Branch, BranchFormat, BranchWriteLockResult,
from ..controldir import ControlDir
from ..decorators import only_raises
from ..lock import LogicalLockResult, _RelockDebugMixin
from ..trace import mutter
from . import bzrdir, rio
from .repository import MetaDirRepository
class BzrBranch(Branch, _RelockDebugMixin):
    """A branch stored in the actual filesystem.

    Note that it's "local" in the context of the filesystem; it doesn't
    really matter if it's on an nfs/smb/afs/coda/... share, as long as
    it's writable, and can be accessed via the normal filesystem API.

    :ivar _transport: Transport for file operations on this branch's
        control files, typically pointing to the .bzr/branch directory.
    :ivar repository: Repository for this branch.
    :ivar base: The url of the base directory for this branch; the one
        containing the .bzr directory.
    :ivar name: Optional colocated branch name as it exists in the control
        directory.
    """
    repository: Union[MetaDirRepository, 'RemoteRepository']
    controldir: bzrdir.BzrDir

    @property
    def control_transport(self) -> _mod_transport.Transport:
        return self._transport

    def __init__(self, *, a_controldir: bzrdir.BzrDir, name: str, _repository: MetaDirRepository, _control_files: lockable_files.LockableFiles, _format=None, ignore_fallbacks=False, possible_transports=None):
        """Create new branch object at a particular location."""
        self.controldir = a_controldir
        self._user_transport = self.controldir.transport.clone('..')
        if name != '':
            self._user_transport.set_segment_parameter('branch', urlutils.escape(name))
        self._base = self._user_transport.base
        self.name = name
        self._format = _format
        self.control_files = _control_files
        self._transport = _control_files._transport
        self.repository = _repository
        self.conf_store = None
        Branch.__init__(self, possible_transports)
        self._tags_bytes = None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.user_url)
    __repr__ = __str__

    def _get_base(self):
        """Returns the directory containing the control directory."""
        return self._base
    base = property(_get_base, doc='The URL for the root of this branch.')

    @property
    def user_transport(self):
        return self._user_transport

    def _get_config(self):
        """Get the concrete config for just the config in this branch.

        This is not intended for client use; see Branch.get_config for the
        public API.

        Added in 1.14.

        :return: An object supporting get_option and set_option.
        """
        return _mod_config.TransportConfig(self._transport, 'branch.conf')

    def _get_config_store(self):
        if self.conf_store is None:
            self.conf_store = _mod_config.BranchStore(self)
        return self.conf_store

    def _uncommitted_branch(self):
        """Return the branch that may contain uncommitted changes."""
        master = self.get_master_branch()
        if master is not None:
            return master
        else:
            return self

    def store_uncommitted(self, creator):
        """Store uncommitted changes from a ShelfCreator.

        :param creator: The ShelfCreator containing uncommitted changes, or
            None to delete any stored changes.
        :raises: ChangesAlreadyStored if the branch already has changes.
        """
        branch = self._uncommitted_branch()
        if creator is None:
            branch._transport.delete('stored-transform')
            return
        if branch._transport.has('stored-transform'):
            raise errors.ChangesAlreadyStored
        transform = BytesIO()
        creator.write_shelf(transform)
        transform.seek(0)
        branch._transport.put_file('stored-transform', transform)

    def get_unshelver(self, tree):
        """Return a shelf.Unshelver for this branch and tree.

        :param tree: The tree to use to construct the Unshelver.
        :return: an Unshelver or None if no changes are stored.
        """
        branch = self._uncommitted_branch()
        try:
            transform = branch._transport.get('stored-transform')
        except _mod_transport.NoSuchFile:
            return None
        return shelf.Unshelver.from_tree_and_shelf(tree, transform)

    def is_locked(self) -> bool:
        return self.control_files.is_locked()

    def lock_write(self, token=None):
        """Lock the branch for write operations.

        :param token: A token to permit reacquiring a previously held and
            preserved lock.
        :return: A BranchWriteLockResult.
        """
        if not self.is_locked():
            self._note_lock('w')
            self.repository._warn_if_deprecated(self)
            self.repository.lock_write()
            took_lock = True
        else:
            took_lock = False
        try:
            return BranchWriteLockResult(self.unlock, self.control_files.lock_write(token=token))
        except BaseException:
            if took_lock:
                self.repository.unlock()
            raise

    def lock_read(self):
        """Lock the branch for read operations.

        :return: A breezy.lock.LogicalLockResult.
        """
        if not self.is_locked():
            self._note_lock('r')
            self.repository._warn_if_deprecated(self)
            self.repository.lock_read()
            took_lock = True
        else:
            took_lock = False
        try:
            self.control_files.lock_read()
            return LogicalLockResult(self.unlock)
        except BaseException:
            if took_lock:
                self.repository.unlock()
            raise

    @only_raises(errors.LockNotHeld, errors.LockBroken)
    def unlock(self):
        if self.control_files._lock_count == 1 and self.conf_store is not None:
            self.conf_store.save_changes()
        try:
            self.control_files.unlock()
        finally:
            if not self.control_files.is_locked():
                self.repository.unlock()
                self._clear_cached_state()

    def peek_lock_mode(self):
        if self.control_files._lock_count == 0:
            return None
        else:
            return self.control_files._lock_mode

    def get_physical_lock_status(self):
        return self.control_files.get_physical_lock_status()

    def set_last_revision_info(self, revno, revision_id):
        if not revision_id or not isinstance(revision_id, bytes):
            raise errors.InvalidRevisionId(revision_id=revision_id, branch=self)
        with self.lock_write():
            old_revno, old_revid = self.last_revision_info()
            if self.get_append_revisions_only():
                self._check_history_violation(revision_id)
            self._run_pre_change_branch_tip_hooks(revno, revision_id)
            self._write_last_revision_info(revno, revision_id)
            self._clear_cached_state()
            self._last_revision_info_cache = (revno, revision_id)
            self._run_post_change_branch_tip_hooks(old_revno, old_revid)

    def basis_tree(self):
        """See Branch.basis_tree."""
        return self.repository.revision_tree(self.last_revision())

    def _get_parent_location(self):
        _locs = ['parent', 'pull', 'x-pull']
        for l in _locs:
            try:
                contents = self._transport.get_bytes(l)
            except _mod_transport.NoSuchFile:
                pass
            else:
                return contents.strip(b'\n').decode('utf-8')
        return None

    def get_stacked_on_url(self):
        raise UnstackableBranchFormat(self._format, self.user_url)

    def set_push_location(self, location):
        """See Branch.set_push_location."""
        self.get_config().set_user_option('push_location', location, store=_mod_config.STORE_LOCATION_NORECURSE)

    def _set_parent_location(self, url):
        if url is None:
            self._transport.delete('parent')
        else:
            if isinstance(url, str):
                url = url.encode('utf-8')
            self._transport.put_bytes('parent', url + b'\n', mode=self.controldir._get_file_mode())

    def unbind(self):
        """If bound, unbind"""
        with self.lock_write():
            return self.set_bound_location(None)

    def bind(self, other):
        """Bind this branch to the branch other.

        This does not push or pull data between the branches, though it does
        check for divergence to raise an error when the branches are not
        either the same, or one a prefix of the other. That behaviour may not
        be useful, so that check may be removed in future.

        :param other: The branch to bind to
        :type other: Branch
        """
        with self.lock_write():
            self.set_bound_location(other.base)

    def get_bound_location(self):
        try:
            return self._transport.get_bytes('bound')[:-1].decode('utf-8')
        except _mod_transport.NoSuchFile:
            return None

    def get_master_branch(self, possible_transports=None):
        """Return the branch we are bound to.

        :return: Either a Branch, or None
        """
        with self.lock_read():
            if self._master_branch_cache is None:
                self._master_branch_cache = self._get_master_branch(possible_transports)
            return self._master_branch_cache

    def _get_master_branch(self, possible_transports):
        bound_loc = self.get_bound_location()
        if not bound_loc:
            return None
        try:
            return Branch.open(bound_loc, possible_transports=possible_transports)
        except (errors.NotBranchError, errors.ConnectionError) as exc:
            raise errors.BoundBranchConnectionFailure(self, bound_loc, exc) from exc

    def set_bound_location(self, location):
        """Set the target where this branch is bound to.

        :param location: URL to the target branch
        """
        with self.lock_write():
            self._master_branch_cache = None
            if location:
                self._transport.put_bytes('bound', location.encode('utf-8') + b'\n', mode=self.controldir._get_file_mode())
            else:
                try:
                    self._transport.delete('bound')
                except _mod_transport.NoSuchFile:
                    return False
                return True

    def update(self, possible_transports=None):
        """Synchronise this branch with the master branch if any.

        :return: None or the last_revision that was pivoted out during the
                 update.
        """
        with self.lock_write():
            master = self.get_master_branch(possible_transports)
            if master is not None:
                old_tip = self.last_revision()
                self.pull(master, overwrite=True)
                if self.repository.get_graph().is_ancestor(old_tip, self.last_revision()):
                    return None
                return old_tip
            return None

    def _read_last_revision_info(self):
        revision_string = self._transport.get_bytes('last-revision')
        revno, revision_id = revision_string.rstrip(b'\n').split(b' ', 1)
        revision_id = cache_utf8.get_cached_utf8(revision_id)
        revno = int(revno)
        return (revno, revision_id)

    def _write_last_revision_info(self, revno, revision_id):
        """Simply write out the revision id, with no checks.

        Use set_last_revision_info to perform this safely.

        Does not update the revision_history cache.
        """
        out_string = b'%d %s\n' % (revno, revision_id)
        self._transport.put_bytes('last-revision', out_string, mode=self.controldir._get_file_mode())

    def update_feature_flags(self, updated_flags):
        """Update the feature flags for this branch.

        :param updated_flags: Dictionary mapping feature names to necessities
            A necessity can be None to indicate the feature should be removed
        """
        with self.lock_write():
            self._format._update_feature_flags(updated_flags)
            self.control_transport.put_bytes('format', self._format.as_string())

    def _get_tags_bytes(self):
        """Get the bytes of a serialised tags dict.

        Note that not all branches support tags, nor do all use the same tags
        logic: this method is specific to BasicTags. Other tag implementations
        may use the same method name and behave differently, safely, because
        of the double-dispatch via
        format.make_tags->tags_instance->get_tags_dict.

        :return: The bytes of the tags file.
        :seealso: Branch._set_tags_bytes.
        """
        with self.lock_read():
            if self._tags_bytes is None:
                self._tags_bytes = self._transport.get_bytes('tags')
            return self._tags_bytes

    def _set_tags_bytes(self, bytes):
        """Mirror method for _get_tags_bytes.

        :seealso: Branch._get_tags_bytes.
        """
        with self.lock_write():
            self._tags_bytes = bytes
            return self._transport.put_bytes('tags', bytes)

    def _clear_cached_state(self):
        super()._clear_cached_state()
        self._tags_bytes = None

    def reconcile(self, thorough=True):
        """Make sure the data stored in this branch is consistent."""
        from .reconcile import BranchReconciler
        with self.lock_write():
            reconciler = BranchReconciler(self, thorough=thorough)
            return reconciler.reconcile()

    def set_reference_info(self, file_id, branch_location, path=None):
        """Set the branch location to use for a tree reference."""
        raise errors.UnsupportedOperation(self.set_reference_info, self)

    def get_reference_info(self, file_id, path=None):
        """Get the tree_path and branch_location for a tree reference."""
        raise errors.UnsupportedOperation(self.get_reference_info, self)

    def reference_parent(self, file_id, path, possible_transports=None):
        """Return the parent branch for a tree-reference.

        :param path: The path of the nested tree in the tree
        :return: A branch associated with the nested tree
        """
        try:
            branch_location = self.get_reference_info(file_id)[0]
        except errors.UnsupportedOperation:
            branch_location = None
        if branch_location is None:
            try:
                return Branch.open_from_transport(self.controldir.root_transport.clone(path), possible_transports=possible_transports)
            except errors.NotBranchError:
                return None
        return Branch.open(urlutils.join(urlutils.strip_segment_parameters(self.user_url), branch_location), possible_transports=possible_transports)

    def set_stacked_on_url(self, url: str) -> None:
        """Set the URL this branch is stacked against.

        :raises UnstackableBranchFormat: If the branch does not support
            stacking.
        :raises UnstackableRepositoryFormat: If the repository does not support
            stacking.
        """
        if not self._format.supports_stacking():
            raise UnstackableBranchFormat(self._format, self.user_url)
        with self.lock_write():
            self._check_stackable_repo()
            if not url:
                try:
                    self.get_stacked_on_url()
                except (errors.NotStacked, UnstackableBranchFormat, errors.UnstackableRepositoryFormat):
                    return
                self._unstack()
            else:
                self._activate_fallback_location(url, possible_transports=[self.controldir.root_transport])
            self._set_config_location('stacked_on_location', url)

    def _check_stackable_repo(self) -> None:
        if not self.repository._format.supports_external_lookups:
            raise errors.UnstackableRepositoryFormat(self.repository._format, self.repository.user_url)

    def _unstack(self):
        """Change a branch to be unstacked, copying data as needed.

        Don't call this directly, use set_stacked_on_url(None).
        """
        with ui.ui_factory.nested_progress_bar() as pb:
            old_repository = self.repository
            if len(old_repository._fallback_repositories) != 1:
                raise AssertionError("can't cope with fallback repositories of %r (fallbacks: %r)" % (old_repository, old_repository._fallback_repositories))
            new_bzrdir = ControlDir.open(self.controldir.root_transport.base)
            new_repository = new_bzrdir.find_repository()
            if new_repository._fallback_repositories:
                raise AssertionError("didn't expect %r to have fallback_repositories" % (self.repository,))
            lock_token = old_repository.lock_write().repository_token
            self.repository = new_repository
            self.repository.lock_write(token=lock_token)
            if lock_token is not None:
                old_repository.leave_lock_in_place()
            old_repository.unlock()
            if lock_token is not None:
                self.repository.dont_leave_lock_in_place()
            old_lock_count = 0
            while True:
                try:
                    old_repository.unlock()
                except errors.LockNotHeld:
                    break
                old_lock_count += 1
            if old_lock_count == 0:
                raise AssertionError('old_repository should have been locked at least once.')
            for i in range(old_lock_count - 1):
                self.repository.lock_write()
            with old_repository.lock_read():
                try:
                    tags_to_fetch = set(self.tags.get_reverse_tag_dict())
                except errors.TagsNotSupported:
                    tags_to_fetch = set()
                fetch_spec = vf_search.NotInOtherForRevs(self.repository, old_repository, required_ids=[self.last_revision()], if_present_ids=tags_to_fetch, find_ghosts=True).execute()
                self.repository.fetch(old_repository, fetch_spec=fetch_spec)

    def break_lock(self) -> None:
        """Break a lock if one is present from another instance.

        Uses the ui factory to ask for confirmation if the lock may be from
        an active process.

        This will probe the repository for its lock as well.
        """
        self.control_files.break_lock()
        self.repository.break_lock()
        master = self.get_master_branch()
        if master is not None:
            master.break_lock()