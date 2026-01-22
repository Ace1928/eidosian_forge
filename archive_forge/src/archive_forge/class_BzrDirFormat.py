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
class BzrDirFormat(BzrFormat, controldir.ControlDirFormat):
    """ControlDirFormat base class for .bzr/ directories.

    Formats are placed in a dict by their format string for reference
    during bzrdir opening. These should be subclasses of BzrDirFormat
    for consistency.

    Once a format is deprecated, just deprecate the initialize and open
    methods on the format class. Do not deprecate the object, as the
    object will be created every system load.
    """
    _lock_file_name = 'branch-lock'

    def initialize_on_transport(self, transport):
        """Initialize a new bzrdir in the base directory of a Transport."""
        try:
            client_medium = transport.get_smart_medium()
        except errors.NoSmartMedium:
            return self._initialize_on_transport_vfs(transport)
        else:
            if not isinstance(self, BzrDirMetaFormat1):
                return self._initialize_on_transport_vfs(transport)
            from .remote import RemoteBzrDirFormat
            remote_format = RemoteBzrDirFormat()
            self._supply_sub_formats_to(remote_format)
            return remote_format.initialize_on_transport(transport)

    def initialize_on_transport_ex(self, transport, use_existing_dir=False, create_prefix=False, force_new_repo=False, stacked_on=None, stack_on_pwd=None, repo_format_name=None, make_working_trees=None, shared_repo=False, vfs_only=False):
        """Create this format on transport.

        The directory to initialize will be created.

        :param force_new_repo: Do not use a shared repository for the target,
                               even if one is available.
        :param create_prefix: Create any missing directories leading up to
            to_transport.
        :param use_existing_dir: Use an existing directory if one exists.
        :param stacked_on: A url to stack any created branch on, None to follow
            any target stacking policy.
        :param stack_on_pwd: If stack_on is relative, the location it is
            relative to.
        :param repo_format_name: If non-None, a repository will be
            made-or-found. Should none be found, or if force_new_repo is True
            the repo_format_name is used to select the format of repository to
            create.
        :param make_working_trees: Control the setting of make_working_trees
            for a new shared repository when one is made. None to use whatever
            default the format has.
        :param shared_repo: Control whether made repositories are shared or
            not.
        :param vfs_only: If True do not attempt to use a smart server
        :return: repo, controldir, require_stacking, repository_policy. repo is
            None if none was created or found, bzrdir is always valid.
            require_stacking is the result of examining the stacked_on
            parameter and any stacking policy found for the target.
        """
        if not vfs_only:
            try:
                client_medium = transport.get_smart_medium()
            except errors.NoSmartMedium:
                pass
            else:
                from .remote import RemoteBzrDirFormat
                remote_dir_format = RemoteBzrDirFormat()
                remote_dir_format._network_name = self.network_name()
                self._supply_sub_formats_to(remote_dir_format)
                return remote_dir_format.initialize_on_transport_ex(transport, use_existing_dir=use_existing_dir, create_prefix=create_prefix, force_new_repo=force_new_repo, stacked_on=stacked_on, stack_on_pwd=stack_on_pwd, repo_format_name=repo_format_name, make_working_trees=make_working_trees, shared_repo=shared_repo)

        def make_directory(transport):
            transport.mkdir('.')
            return transport

        def redirected(transport, e, redirection_notice):
            note(redirection_notice)
            return transport._redirected_to(e.source, e.target)
        try:
            transport = do_catching_redirections(make_directory, transport, redirected)
        except _mod_transport.FileExists:
            if not use_existing_dir:
                raise
        except _mod_transport.NoSuchFile:
            if not create_prefix:
                raise
            transport.create_prefix()
        require_stacking = stacked_on is not None
        result = self.initialize_on_transport(transport)
        if repo_format_name:
            try:
                result._format.repository_format = repository.network_format_registry.get(repo_format_name)
            except AttributeError:
                pass
            repository_policy = result.determine_repository_policy(force_new_repo, stacked_on, stack_on_pwd, require_stacking=require_stacking)
            result_repo, is_new_repo = repository_policy.acquire_repository(make_working_trees, shared_repo)
            if not require_stacking and repository_policy._require_stacking:
                require_stacking = True
                result._format.require_stacking()
            result_repo.lock_write()
        else:
            result_repo = None
            repository_policy = None
        return (result_repo, result, require_stacking, repository_policy)

    def _initialize_on_transport_vfs(self, transport):
        """Initialize a new bzrdir using VFS calls.

        :param transport: The transport to create the .bzr directory in.
        :return: A
        """
        temp_control = lockable_files.LockableFiles(transport, '', lockable_files.TransportLock)
        try:
            temp_control._transport.mkdir('.bzr', mode=temp_control._dir_mode)
        except _mod_transport.FileExists:
            raise errors.AlreadyControlDirError(transport.base)
        if sys.platform == 'win32' and isinstance(transport, local.LocalTransport):
            win32utils.set_file_attr_hidden(transport._abspath('.bzr'))
        file_mode = temp_control._file_mode
        del temp_control
        bzrdir_transport = transport.clone('.bzr')
        utf8_files = [('README', b'This is a Bazaar control directory.\nDo not change any files in this directory.\nSee http://bazaar.canonical.com/ for more information about Bazaar.\n'), ('branch-format', self.as_string())]
        control_files = lockable_files.LockableFiles(bzrdir_transport, self._lock_file_name, self._lock_class)
        control_files.create_lock()
        control_files.lock_write()
        try:
            for filename, content in utf8_files:
                bzrdir_transport.put_bytes(filename, content, mode=file_mode)
        finally:
            control_files.unlock()
        return self.open(transport, _found=True)

    def open(self, transport, _found=False):
        """Return an instance of this format for the dir transport points at.

        _found is a private parameter, do not use it.
        """
        if not _found:
            found_format = controldir.ControlDirFormat.find_format(transport)
            if not isinstance(found_format, self.__class__):
                raise AssertionError('%s was asked to open %s, but it seems to need format %s' % (self, transport, found_format))
            self._supply_sub_formats_to(found_format)
            return found_format._open(transport)
        return self._open(transport)

    def _open(self, transport):
        """Template method helper for opening BzrDirectories.

        This performs the actual open and any additional logic or parameter
        passing.
        """
        raise NotImplementedError(self._open)

    def _supply_sub_formats_to(self, other_format):
        """Give other_format the same values for sub formats as this has.

        This method is expected to be used when parameterising a
        RemoteBzrDirFormat instance with the parameters from a
        BzrDirMetaFormat1 instance.

        :param other_format: other_format is a format which should be
            compatible with whatever sub formats are supported by self.
        :return: None.
        """
        other_format.features = dict(self.features)

    def supports_transport(self, transport):
        return True

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        controldir.ControlDirFormat.check_support_status(self, allow_unsupported=allow_unsupported, recommend_upgrade=recommend_upgrade, basedir=basedir)
        BzrFormat.check_support_status(self, allow_unsupported=allow_unsupported, recommend_upgrade=recommend_upgrade, basedir=basedir)

    @classmethod
    def is_control_filename(klass, filename):
        """True if filename is the name of a path which is reserved for bzrdir's.

        :param filename: A filename within the root transport of this bzrdir.

        This is true IF and ONLY IF the filename is part of the namespace
        reserved for bzr control dirs. Currently this is the '.bzr' directory
        in the root of the root_transport.
        """
        return filename == '.bzr' or filename.startswith('.bzr/')

    @classmethod
    def get_default_format(klass):
        """Return the current default format."""
        return controldir.format_registry.get('bzr')()