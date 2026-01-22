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
class LocalGitDir(GitDir):
    """An adapter to the '.git' dir used by git."""

    def _get_gitrepository_class(self):
        from .repository import LocalGitRepository
        return LocalGitRepository

    def __repr__(self):
        return '<{} at {!r}>'.format(self.__class__.__name__, self.root_transport.base)
    _gitrepository_class = property(_get_gitrepository_class)

    @property
    def user_transport(self):
        return self.root_transport

    @property
    def control_transport(self):
        return self._git._controltransport

    def __init__(self, transport, gitrepo, format):
        self._format = format
        self.root_transport = transport
        self._mode_check_done = False
        self._git = gitrepo
        if gitrepo.bare:
            self.transport = transport
        else:
            self.transport = transport.clone('.git')
        self._mode_check_done = None

    def _get_symref(self, ref):
        ref_chain, unused_sha = self._git.refs.follow(ref)
        if len(ref_chain) == 1:
            return None
        return ref_chain[1]

    def set_branch_reference(self, target_branch, name=None):
        ref = self._get_selected_ref(name)
        target_transport = target_branch.controldir.control_transport
        if self.control_transport.base == target_transport.base:
            if ref == target_branch.ref:
                raise BranchReferenceLoop(target_branch)
            self._git.refs.set_symbolic_ref(ref, target_branch.ref)
        else:
            try:
                target_path = target_branch.controldir.control_transport.local_abspath('.')
            except brz_errors.NotLocalUrl:
                raise brz_errors.IncompatibleFormat(target_branch._format, self._format)
            self.control_transport.put_bytes('commondir', encode_git_path(target_path))
            self._git._commontransport = target_branch.repository._git._commontransport.clone()
            self._git.object_store = TransportObjectStore(self._git._commontransport.clone(OBJECTDIR))
            self._git.refs.transport = self._git._commontransport
            target_ref_chain, unused_sha = target_branch.controldir._git.refs.follow(target_branch.ref)
            for target_ref in target_ref_chain:
                if target_ref == b'HEAD':
                    continue
                break
            else:
                raise brz_errors.IncompatibleFormat(self.set_branch_reference, self)
            self._git.refs.set_symbolic_ref(ref, target_ref)

    def get_branch_reference(self, name=None):
        ref = self._get_selected_ref(name)
        try:
            target_ref = self._get_symref(ref)
        except SymrefLoop:
            raise BranchReferenceLoop(self)
        if target_ref is not None:
            from .refs import ref_to_branch_name
            try:
                branch_name = ref_to_branch_name(target_ref)
            except ValueError:
                params = {'ref': urlutils.quote(target_ref.decode('utf-8'), '')}
            else:
                if branch_name != '':
                    params = {'branch': urlutils.quote(branch_name, '')}
                else:
                    params = {}
            try:
                commondir = self.control_transport.get_bytes('commondir')
            except NoSuchFile:
                base_url = self.user_url.rstrip('/')
            else:
                base_url = urlutils.local_path_to_url(decode_git_path(commondir)).rstrip('/.git/') + '/'
            return urlutils.join_segment_parameters(base_url, params)
        return None

    def find_branch_format(self, name=None):
        from .branch import LocalGitBranchFormat
        return LocalGitBranchFormat()

    def get_branch_transport(self, branch_format, name=None):
        if branch_format is None:
            return self.transport
        if isinstance(branch_format, LocalGitControlDirFormat):
            return self.transport
        raise brz_errors.IncompatibleFormat(branch_format, self._format)

    def get_repository_transport(self, format):
        if format is None:
            return self.transport
        if isinstance(format, LocalGitControlDirFormat):
            return self.transport
        raise brz_errors.IncompatibleFormat(format, self._format)

    def get_workingtree_transport(self, format):
        if format is None:
            return self.transport
        if isinstance(format, LocalGitControlDirFormat):
            return self.transport
        raise brz_errors.IncompatibleFormat(format, self._format)

    def open_branch(self, name=None, unsupported=False, ignore_fallbacks=None, ref=None, possible_transports=None, nascent_ok=False):
        """'create' a branch for this dir."""
        repo = self.find_repository()
        from .branch import LocalGitBranch
        ref = self._get_selected_ref(name, ref)
        if not nascent_ok and ref not in self._git.refs:
            raise brz_errors.NotBranchError(self.root_transport.base, controldir=self)
        try:
            ref_chain, unused_sha = self._git.refs.follow(ref)
        except SymrefLoop as e:
            raise BranchReferenceLoop(self)
        if ref_chain[-1] == b'HEAD':
            controldir = self
        else:
            controldir = self._find_commondir()
        return LocalGitBranch(controldir, repo, ref_chain[-1])

    def destroy_branch(self, name=None):
        refname = self._get_selected_ref(name)
        if refname == b'HEAD':
            raise brz_errors.UnsupportedOperation(self.destroy_branch, self)
        try:
            del self._git.refs[refname]
        except KeyError:
            raise brz_errors.NotBranchError(self.root_transport.base, controldir=self)

    def destroy_repository(self):
        raise brz_errors.UnsupportedOperation(self.destroy_repository, self)

    def destroy_workingtree(self):
        raise brz_errors.UnsupportedOperation(self.destroy_workingtree, self)

    def destroy_workingtree_metadata(self):
        raise brz_errors.UnsupportedOperation(self.destroy_workingtree_metadata, self)

    def needs_format_conversion(self, format=None):
        return not isinstance(self._format, format.__class__)

    def open_repository(self):
        """'open' a repository for this dir."""
        if self.control_transport.has('commondir'):
            raise brz_errors.NoRepositoryPresent(self)
        return self._gitrepository_class(self)

    def has_workingtree(self):
        return not self._git.bare

    def open_workingtree(self, recommend_upgrade=True, unsupported=False):
        if not self._git.bare:
            repo = self.find_repository()
            from .workingtree import GitWorkingTree
            branch = self.open_branch(ref=b'HEAD', nascent_ok=True)
            return GitWorkingTree(self, repo, branch)
        loc = urlutils.unescape_for_display(self.root_transport.base, 'ascii')
        raise brz_errors.NoWorkingTree(loc)

    def create_repository(self, shared=False):
        from .repository import GitRepositoryFormat
        if shared:
            raise brz_errors.IncompatibleFormat(GitRepositoryFormat(), self._format)
        return self.find_repository()

    def create_branch(self, name=None, repository=None, append_revisions_only=None, ref=None):
        refname = self._get_selected_ref(name, ref)
        if refname != b'HEAD' and refname in self._git.refs:
            raise brz_errors.AlreadyBranchError(self.user_url)
        repo = self.open_repository()
        if refname in self._git.refs:
            ref_chain, unused_sha = self._git.refs.follow(self._get_selected_ref(None))
            if ref_chain[0] == b'HEAD':
                refname = ref_chain[1]
        from .branch import LocalGitBranch
        branch = LocalGitBranch(self, repo, refname)
        if append_revisions_only:
            branch.set_append_revisions_only(append_revisions_only)
        return branch

    def backup_bzrdir(self):
        if not self._git.bare:
            self.root_transport.copy_tree('.git', '.git.backup')
            return (self.root_transport.abspath('.git'), self.root_transport.abspath('.git.backup'))
        else:
            basename = urlutils.basename(self.root_transport.base)
            parent = self.root_transport.clone('..')
            parent.copy_tree(basename, basename + '.backup')

    def create_workingtree(self, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False):
        if self._git.bare:
            raise brz_errors.UnsupportedOperation(self.create_workingtree, self)
        if from_branch is None:
            from_branch = self.open_branch(nascent_ok=True)
        if revision_id is None:
            revision_id = from_branch.last_revision()
        repo = self.find_repository()
        from .workingtree import GitWorkingTree
        wt = GitWorkingTree(self, repo, from_branch)
        wt.set_last_revision(revision_id)
        wt._build_checkout_with_index()
        return wt

    def _find_or_create_repository(self, force_new_repo=None):
        return self.create_repository(shared=False)

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
        except brz_errors.TransportNotPossible:
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

    def get_refs_container(self):
        return self._git.refs

    def get_peeled(self, ref):
        return self._git.get_peeled(ref)

    def _find_commondir(self):
        try:
            commondir = self.control_transport.get_bytes('commondir')
        except NoSuchFile:
            return self
        else:
            commondir = os.fsdecode(commondir.rstrip(b'/.git/'))
            return ControlDir.open_from_transport(get_transport_from_path(commondir))