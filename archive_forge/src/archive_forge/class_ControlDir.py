from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class ControlDir(ControlComponent):
    """A control directory.

    While this represents a generic control directory, there are a few
    features that are present in this interface that are currently only
    supported by one of its implementations, BzrDir.

    These features (bound branches, stacked branches) are currently only
    supported by Bazaar, but could be supported by other version control
    systems as well. Implementations are required to raise the appropriate
    exceptions when an operation is requested that is not supported.

    This also makes life easier for API users who can rely on the
    implementation always allowing a particular feature to be requested but
    raising an exception when it is not supported, rather than requiring the
    API users to check for magic attributes to see what features are supported.
    """
    hooks: hooks.Hooks
    root_transport: _mod_transport.Transport
    user_transport: _mod_transport.Transport

    def can_convert_format(self):
        """Return true if this controldir is one whose format we can convert
        from."""
        return True

    def list_branches(self) -> List['Branch']:
        """Return a sequence of all branches local to this control directory.

        """
        return list(self.get_branches().values())

    def branch_names(self) -> List[str]:
        """List all branch names in this control directory.

        Returns: List of branch names
        """
        try:
            self.get_branch_reference()
        except (errors.NotBranchError, errors.NoRepositoryPresent):
            return []
        else:
            return ['']

    def get_branches(self) -> Dict[str, 'Branch']:
        """Get all branches in this control directory, as a dictionary.

        Returns: Dictionary mapping branch names to instances.
        """
        try:
            return {'': self.open_branch()}
        except (errors.NotBranchError, errors.NoRepositoryPresent):
            return {}

    def is_control_filename(self, filename):
        """True if filename is the name of a path which is reserved for
        controldirs.

        Args:
          filename: A filename within the root transport of this
            controldir.

        This is true IF and ONLY IF the filename is part of the namespace reserved
        for bzr control dirs. Currently this is the '.bzr' directory in the root
        of the root_transport. it is expected that plugins will need to extend
        this in the future - for instance to make bzr talk with svn working
        trees.
        """
        return self._format.is_control_filename(filename)

    def needs_format_conversion(self, format=None):
        """Return true if this controldir needs convert_format run on it.

        For instance, if the repository format is out of date but the
        branch and working tree are not, this should return True.

        Args:
          format: Optional parameter indicating a specific desired
                       format we plan to arrive at.
        """
        raise NotImplementedError(self.needs_format_conversion)

    def create_repository(self, shared: bool=False) -> 'Repository':
        """Create a new repository in this control directory.

        Args:
          shared: If a shared repository should be created

        Returns: The newly created repository
        """
        raise NotImplementedError(self.create_repository)

    def destroy_repository(self) -> None:
        """Destroy the repository in this ControlDir."""
        raise NotImplementedError(self.destroy_repository)

    def create_branch(self, name: Optional[str]=None, repository: Optional['Repository']=None, append_revisions_only: Optional[bool]=None) -> 'Branch':
        """Create a branch in this ControlDir.

        Args:
          name: Name of the colocated branch to create, None for
            the user selected branch or "" for the active branch.
          append_revisions_only: Whether this branch should only allow
            appending new revisions to its history.

        The controldirs format will control what branch format is created.
        For more control see BranchFormatXX.create(a_controldir).
        """
        raise NotImplementedError(self.create_branch)

    def destroy_branch(self, name: Optional[str]=None) -> None:
        """Destroy a branch in this ControlDir.

        Args:
          name: Name of the branch to destroy, None for the
            user selected branch or "" for the active branch.

        Raises:
          NotBranchError: When the branch does not exist
        """
        raise NotImplementedError(self.destroy_branch)

    def create_workingtree(self, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False) -> 'WorkingTree':
        """Create a working tree at this ControlDir.

        Args:
          revision_id: create it as of this revision id.
          from_branch: override controldir branch
            (for lightweight checkouts)
          accelerator_tree: A tree which can be used for retrieving file
            contents more quickly than the revision tree, i.e. a workingtree.
            The revision tree will be used for cases where accelerator_tree's
            content is different.
        """
        raise NotImplementedError(self.create_workingtree)

    def destroy_workingtree(self):
        """Destroy the working tree at this ControlDir.

        Formats that do not support this may raise UnsupportedOperation.
        """
        raise NotImplementedError(self.destroy_workingtree)

    def destroy_workingtree_metadata(self):
        """Destroy the control files for the working tree at this ControlDir.

        The contents of working tree files are not affected.
        Formats that do not support this may raise UnsupportedOperation.
        """
        raise NotImplementedError(self.destroy_workingtree_metadata)

    def find_branch_format(self, name=None):
        """Find the branch 'format' for this controldir.

        This might be a synthetic object for e.g. RemoteBranch and SVN.
        """
        raise NotImplementedError(self.find_branch_format)

    def get_branch_reference(self, name=None):
        """Return the referenced URL for the branch in this controldir.

        Args:
          name: Optional colocated branch name

        Raises:
          NotBranchError: If there is no Branch.
          NoColocatedBranchSupport: If a branch name was specified
            but colocated branches are not supported.

        Returns:
          The URL the branch in this controldir references if it is a
          reference branch, or None for regular branches.
        """
        if name is not None:
            raise NoColocatedBranchSupport(self)
        return None

    def set_branch_reference(self, target_branch, name=None):
        """Set the referenced URL for the branch in this controldir.

        Args:
          name: Optional colocated branch name
          target_branch: Branch to reference

        Raises:
          NoColocatedBranchSupport: If a branch name was specified
            but colocated branches are not supported.

        Returns:
          The referencing branch
        """
        raise NotImplementedError(self.set_branch_reference)

    def open_branch(self, name=None, unsupported=False, ignore_fallbacks=False, possible_transports=None) -> 'Branch':
        """Open the branch object at this ControlDir if one is present.

        Args:
          unsupported: if True, then no longer supported branch formats can
            still be opened.
          ignore_fallbacks: Whether to open fallback repositories
          possible_transports: Transports to use for opening e.g.
            fallback repositories.
        """
        raise NotImplementedError(self.open_branch)

    def open_repository(self, _unsupported=False) -> 'Repository':
        """Open the repository object at this ControlDir if one is present.

        This will not follow the Branch object pointer - it's strictly a direct
        open facility. Most client code should use open_branch().repository to
        get at a repository.

        Args:
          _unsupported: a private parameter, not part of the api.
        """
        raise NotImplementedError(self.open_repository)

    def find_repository(self) -> 'Repository':
        """Find the repository that should be used.

        This does not require a branch as we use it to find the repo for
        new branches as well as to hook existing branches up to their
        repository.
        """
        raise NotImplementedError(self.find_repository)

    def open_workingtree(self, unsupported=False, recommend_upgrade=True, from_branch=None) -> 'WorkingTree':
        """Open the workingtree object at this ControlDir if one is present.

        Args:
          recommend_upgrade: Optional keyword parameter, when True (the
            default), emit through the ui module a recommendation that the user
            upgrade the working tree when the workingtree being opened is old
            (but still fully supported).
          from_branch: override controldir branch (for lightweight
            checkouts)
        """
        raise NotImplementedError(self.open_workingtree)

    def has_branch(self, name=None):
        """Tell if this controldir contains a branch.

        Note: if you're going to open the branch, you should just go ahead
        and try, and not ask permission first.  (This method just opens the
        branch and discards it, and that's somewhat expensive.)
        """
        try:
            self.open_branch(name, ignore_fallbacks=True)
            return True
        except errors.NotBranchError:
            return False

    def _get_selected_branch(self):
        """Return the name of the branch selected by the user.

        Returns: Name of the branch selected by the user, or "".
        """
        branch = self.root_transport.get_segment_parameters().get('branch')
        if branch is None:
            branch = ''
        return urlutils.unescape(branch)

    def has_workingtree(self):
        """Tell if this controldir contains a working tree.

        This will still raise an exception if the controldir has a workingtree
        that is remote & inaccessible.

        Note: if you're going to open the working tree, you should just go ahead
        and try, and not ask permission first.  (This method just opens the
        workingtree and discards it, and that's somewhat expensive.)
        """
        try:
            self.open_workingtree(recommend_upgrade=False)
            return True
        except errors.NoWorkingTree:
            return False

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
        raise NotImplementedError(self.cloning_metadir)

    def checkout_metadir(self):
        """Produce a metadir suitable for checkouts of this controldir.

        :returns: A ControlDirFormat with all component formats
            either set appropriately or set to None if that component
            should not be created.
        """
        return self.cloning_metadir()

    def sprout(self, url, revision_id=None, force_new_repo=False, recurse='down', possible_transports=None, accelerator_tree=None, hardlink=False, stacked=False, source_branch=None, create_tree_if_local=True, lossy=False):
        """Create a copy of this controldir prepared for use as a new line of
        development.

        If url's last component does not exist, it will be created.

        Attributes related to the identity of the source branch like
        branch nickname will be cleaned, a working tree is created
        whether one existed before or not; and a local branch is always
        created.

        Args:
          revision_id: if revision_id is not None, then the clone
            operation may tune itself to download less data.
          accelerator_tree: A tree which can be used for retrieving file
            contents more quickly than the revision tree, i.e. a workingtree.
            The revision tree will be used for cases where accelerator_tree's
            content is different.
          hardlink: If true, hard-link files from accelerator_tree,
            where possible.
          stacked: If true, create a stacked branch referring to the
            location of this control directory.
          create_tree_if_local: If true, a working-tree will be created
            when working locally.
        """
        raise NotImplementedError(self.sprout)

    def push_branch(self, source, revision_id=None, overwrite=False, remember=False, create_prefix=False, lossy=False, tag_selector=None, name=None):
        """Push the source branch into this ControlDir."""
        from .push import PushResult
        try:
            br_to = self.open_branch(name=name)
        except errors.NotBranchError:
            repository_to = self.find_repository()
            br_to = None
        else:
            repository_to = br_to.repository
        push_result = PushResult()
        push_result.source_branch = source
        if br_to is None:
            if revision_id is None:
                revision_id = source.last_revision()
            repository_to.fetch(source.repository, revision_id=revision_id)
            br_to = source.sprout(self, revision_id=revision_id, lossy=lossy, tag_selector=tag_selector, name=name)
            if source.get_push_location() is None or remember:
                source.set_push_location(br_to.base)
            push_result.stacked_on = None
            push_result.branch_push_result = None
            push_result.old_revno = None
            push_result.old_revid = _mod_revision.NULL_REVISION
            push_result.target_branch = br_to
            push_result.master_branch = None
            push_result.workingtree_updated = False
        else:
            if source.get_push_location() is None or remember:
                source.set_push_location(br_to.base)
            try:
                tree_to = self.open_workingtree()
            except errors.NotLocalUrl:
                push_result.branch_push_result = source.push(br_to, overwrite=overwrite, stop_revision=revision_id, lossy=lossy, tag_selector=tag_selector)
                push_result.workingtree_updated = False
            except errors.NoWorkingTree:
                push_result.branch_push_result = source.push(br_to, overwrite=overwrite, stop_revision=revision_id, lossy=lossy, tag_selector=tag_selector)
                push_result.workingtree_updated = None
            else:
                if br_to.name == tree_to.branch.name:
                    with tree_to.lock_write():
                        push_result.branch_push_result = source.push(tree_to.branch, overwrite=overwrite, stop_revision=revision_id, lossy=lossy, tag_selector=tag_selector)
                        tree_to.update()
                    push_result.workingtree_updated = True
                else:
                    push_result.branch_push_result = source.push(br_to, overwrite=overwrite, stop_revision=revision_id, lossy=lossy, tag_selector=tag_selector)
                    push_result.workingtree_updated = None
            push_result.old_revno = push_result.branch_push_result.old_revno
            push_result.old_revid = push_result.branch_push_result.old_revid
            push_result.target_branch = push_result.branch_push_result.target_branch
        return push_result

    def _get_tree_branch(self, name=None):
        """Return the branch and tree, if any, for this controldir.

        Args:
          name: Name of colocated branch to open.

        Return None for tree if not present or inaccessible.
        Raise NotBranchError if no branch is present.

        Returns: (tree, branch)
        """
        try:
            tree = self.open_workingtree()
        except (errors.NoWorkingTree, errors.NotLocalUrl):
            tree = None
            branch = self.open_branch(name=name)
        else:
            if name is not None:
                branch = self.open_branch(name=name)
            else:
                branch = tree.branch
        return (tree, branch)

    def get_config(self):
        """Get configuration for this ControlDir."""
        raise NotImplementedError(self.get_config)

    def check_conversion_target(self, target_format):
        """Check that a controldir as a whole can be converted to a new format."""
        raise NotImplementedError(self.check_conversion_target)

    def clone(self, url, revision_id=None, force_new_repo=False, preserve_stacking=False, tag_selector=None):
        """Clone this controldir and its contents to url verbatim.

        Args:
          url: The url create the clone at.  If url's last component does
            not exist, it will be created.
          revision_id: The tip revision-id to use for any branch or
            working tree.  If not None, then the clone operation may tune
            itself to download less data.
          force_new_repo: Do not use a shared repository for the target
                               even if one is available.
          preserve_stacking: When cloning a stacked branch, stack the
            new branch on top of the other branch's stacked-on branch.
        """
        return self.clone_on_transport(_mod_transport.get_transport(url), revision_id=revision_id, force_new_repo=force_new_repo, preserve_stacking=preserve_stacking, tag_selector=tag_selector)

    def clone_on_transport(self, transport, revision_id=None, force_new_repo=False, preserve_stacking=False, stacked_on=None, create_prefix=False, use_existing_dir=True, no_tree=False, tag_selector=None):
        """Clone this controldir and its contents to transport verbatim.

        Args:
          transport: The transport for the location to produce the clone
            at.  If the target directory does not exist, it will be created.
          revision_id: The tip revision-id to use for any branch or
            working tree.  If not None, then the clone operation may tune
            itself to download less data.
          force_new_repo: Do not use a shared repository for the target,
                               even if one is available.
          preserve_stacking: When cloning a stacked branch, stack the
            new branch on top of the other branch's stacked-on branch.
          create_prefix: Create any missing directories leading up to
            to_transport.
          use_existing_dir: Use an existing directory if one exists.
          no_tree: If set to true prevents creation of a working tree.
        """
        raise NotImplementedError(self.clone_on_transport)

    @classmethod
    def find_controldirs(klass, transport, evaluate=None, list_current=None):
        """Find control dirs recursively from current location.

        This is intended primarily as a building block for more sophisticated
        functionality, like finding trees under a directory, or finding
        branches that use a given repository.

        Args:
          evaluate: An optional callable that yields recurse, value,
            where recurse controls whether this controldir is recursed into
            and value is the value to yield.  By default, all bzrdirs
            are recursed into, and the return value is the controldir.
          list_current: if supplied, use this function to list the current
            directory, instead of Transport.list_dir

        Returns:
          a generator of found bzrdirs, or whatever evaluate returns.
        """
        if list_current is None:

            def list_current(transport):
                return transport.list_dir('')
        if evaluate is None:

            def evaluate(controldir):
                return (True, controldir)
        pending = [transport]
        while len(pending) > 0:
            current_transport = pending.pop()
            recurse = True
            try:
                controldir = klass.open_from_transport(current_transport)
            except (errors.NotBranchError, errors.PermissionDenied, errors.UnknownFormatError):
                pass
            else:
                recurse, value = evaluate(controldir)
                yield value
            try:
                subdirs = list_current(current_transport)
            except (_mod_transport.NoSuchFile, errors.PermissionDenied):
                continue
            if recurse:
                for subdir in sorted(subdirs, reverse=True):
                    pending.append(current_transport.clone(subdir))

    @classmethod
    def find_branches(klass, transport):
        """Find all branches under a transport.

        This will find all branches below the transport, including branches
        inside other branches.  Where possible, it will use
        Repository.find_branches.

        To list all the branches that use a particular Repository, see
        Repository.find_branches
        """

        def evaluate(controldir):
            try:
                repository = controldir.open_repository()
            except errors.NoRepositoryPresent:
                pass
            else:
                return (False, ([], repository))
            return (True, (controldir.list_branches(), None))
        ret = []
        for branches, repo in klass.find_controldirs(transport, evaluate=evaluate):
            if repo is not None:
                ret.extend(repo.find_branches())
            if branches is not None:
                ret.extend(branches)
        return ret

    @classmethod
    def create_branch_and_repo(klass, base, force_new_repo=False, format=None) -> 'Branch':
        """Create a new ControlDir, Branch and Repository at the url 'base'.

        This will use the current default ControlDirFormat unless one is
        specified, and use whatever
        repository format that that uses via controldir.create_branch and
        create_repository. If a shared repository is available that is used
        preferentially.

        The created Branch object is returned.

        Args:
          base: The URL to create the branch at.
          force_new_repo: If True a new repository is always created.
          format: If supplied, the format of branch to create.  If not
            supplied, the default is used.
        """
        controldir = klass.create(base, format)
        controldir._find_or_create_repository(force_new_repo)
        return cast('Branch', controldir.create_branch())

    @classmethod
    def create_branch_convenience(klass, base, force_new_repo=False, force_new_tree=None, format=None, possible_transports=None):
        """Create a new ControlDir, Branch and Repository at the url 'base'.

        This is a convenience function - it will use an existing repository
        if possible, can be told explicitly whether to create a working tree or
        not.

        This will use the current default ControlDirFormat unless one is
        specified, and use whatever
        repository format that that uses via ControlDir.create_branch and
        create_repository. If a shared repository is available that is used
        preferentially. Whatever repository is used, its tree creation policy
        is followed.

        The created Branch object is returned.
        If a working tree cannot be made due to base not being a file:// url,
        no error is raised unless force_new_tree is True, in which case no
        data is created on disk and NotLocalUrl is raised.

        Args:
          base: The URL to create the branch at.
          force_new_repo: If True a new repository is always created.
          force_new_tree: If True or False force creation of a tree or
                               prevent such creation respectively.
          format: Override for the controldir format to create.
          possible_transports: An optional reusable transports list.
        """
        if force_new_tree:
            from breezy.transport import local
            t = _mod_transport.get_transport(base, possible_transports)
            if not isinstance(t, local.LocalTransport):
                raise errors.NotLocalUrl(base)
        controldir = klass.create(base, format, possible_transports)
        repo = controldir._find_or_create_repository(force_new_repo)
        result = controldir.create_branch()
        if force_new_tree or (repo.make_working_trees() and force_new_tree is None):
            try:
                controldir.create_workingtree()
            except errors.NotLocalUrl:
                pass
        return result

    @classmethod
    def create_standalone_workingtree(klass, base, format=None) -> 'WorkingTree':
        """Create a new ControlDir, WorkingTree, Branch and Repository at 'base'.

        'base' must be a local path or a file:// url.

        This will use the current default ControlDirFormat unless one is
        specified, and use whatever
        repository format that that uses for bzrdirformat.create_workingtree,
        create_branch and create_repository.

        Args:
          format: Override for the controldir format to create.

        Returns: The WorkingTree object.
        """
        t = _mod_transport.get_transport(base)
        from breezy.transport import local
        if not isinstance(t, local.LocalTransport):
            raise errors.NotLocalUrl(base)
        controldir = klass.create_branch_and_repo(base, force_new_repo=True, format=format).controldir
        return controldir.create_workingtree()

    @classmethod
    def open_unsupported(klass, base):
        """Open a branch which is not supported."""
        return klass.open(base, _unsupported=True)

    @classmethod
    def open(klass, base, possible_transports=None, probers=None, _unsupported=False) -> 'ControlDir':
        """Open an existing controldir, rooted at 'base' (url).

        Args:
          _unsupported: a private parameter to the ControlDir class.
        """
        t = _mod_transport.get_transport(base, possible_transports)
        return klass.open_from_transport(t, probers=probers, _unsupported=_unsupported)

    @classmethod
    def open_from_transport(klass, transport: _mod_transport.Transport, _unsupported=False, probers=None) -> 'ControlDir':
        """Open a controldir within a particular directory.

        Args:
          transport: Transport containing the controldir.
          _unsupported: private.
        """
        for hook in klass.hooks['pre_open']:
            hook(transport)
        base = transport.base

        def find_format(transport):
            return (transport, ControlDirFormat.find_format(transport, probers=probers))

        def redirected(transport, e, redirection_notice):
            redirected_transport = transport._redirected_to(e.source, e.target)
            if redirected_transport is None:
                raise errors.NotBranchError(base)
            trace.note(gettext('{0} is{1} redirected to {2}').format(transport.base, e.permanently, redirected_transport.base))
            return redirected_transport
        try:
            transport, format = _mod_transport.do_catching_redirections(find_format, transport, redirected)
        except errors.TooManyRedirections:
            raise errors.NotBranchError(base)
        format.check_support_status(_unsupported)
        return cast('ControlDir', format.open(transport, _found=True))

    @classmethod
    def open_containing(klass, url, possible_transports=None):
        """Open an existing branch which contains url.

        Args:
          url: url to search from.

        See open_containing_from_transport for more detail.
        """
        transport = _mod_transport.get_transport(url, possible_transports)
        return klass.open_containing_from_transport(transport)

    @classmethod
    def open_containing_from_transport(klass, a_transport, probers=None):
        """Open an existing branch which contains a_transport.base.

        This probes for a branch at a_transport, and searches upwards from there.

        Basically we keep looking up until we find the control directory or
        run into the root.  If there isn't one, raises NotBranchError.
        If there is one and it is either an unrecognised format or an unsupported
        format, UnknownFormatError or UnsupportedFormatError are raised.
        If there is one, it is returned, along with the unused portion of url.

        Returns: The ControlDir that contains the path, and a Unicode path
                for the rest of the URL.
        """
        url = a_transport.base
        while True:
            try:
                result = klass.open_from_transport(a_transport, probers=probers)
                return (result, urlutils.unescape(a_transport.relpath(url)))
            except errors.NotBranchError:
                pass
            except errors.PermissionDenied:
                pass
            try:
                new_t = a_transport.clone('..')
            except urlutils.InvalidURLJoin:
                raise errors.NotBranchError(path=url)
            if new_t.base == a_transport.base:
                raise errors.NotBranchError(path=url)
            a_transport = new_t

    @classmethod
    def open_tree_or_branch(klass, location, name=None):
        """Return the branch and working tree at a location.

        If there is no tree at the location, tree will be None.
        If there is no branch at the location, an exception will be
        raised
        Returns: (tree, branch)
        """
        controldir = klass.open(location)
        return controldir._get_tree_branch(name=name)

    @classmethod
    def open_containing_tree_or_branch(klass, location, possible_transports=None):
        """Return the branch and working tree contained by a location.

        Returns (tree, branch, relpath).
        If there is no tree at containing the location, tree will be None.
        If there is no branch containing the location, an exception will be
        raised
        relpath is the portion of the path that is contained by the branch.
        """
        controldir, relpath = klass.open_containing(location, possible_transports=possible_transports)
        tree, branch = controldir._get_tree_branch()
        return (tree, branch, relpath)

    @classmethod
    def open_containing_tree_branch_or_repository(klass, location):
        """Return the working tree, branch and repo contained by a location.

        Returns (tree, branch, repository, relpath).
        If there is no tree containing the location, tree will be None.
        If there is no branch containing the location, branch will be None.
        If there is no repository containing the location, repository will be
        None.
        relpath is the portion of the path that is contained by the innermost
        ControlDir.

        If no tree, branch or repository is found, a NotBranchError is raised.
        """
        controldir, relpath = klass.open_containing(location)
        try:
            tree, branch = controldir._get_tree_branch()
        except errors.NotBranchError:
            try:
                repo = controldir.find_repository()
                return (None, None, repo, relpath)
            except errors.NoRepositoryPresent:
                raise errors.NotBranchError(location)
        return (tree, branch, branch.repository, relpath)

    @classmethod
    def create(klass, base, format=None, possible_transports=None):
        """Create a new ControlDir at the url 'base'.

        Args:
          format: If supplied, the format of branch to create.  If not
            supplied, the default is used.
          possible_transports: If supplied, a list of transports that
            can be reused to share a remote connection.
        """
        if klass is not ControlDir:
            raise AssertionError('ControlDir.create always creates thedefault format, not one of %r' % klass)
        t = _mod_transport.get_transport(base, possible_transports)
        t.ensure_base()
        if format is None:
            format = ControlDirFormat.get_default_format()
        return format.initialize_on_transport(t)