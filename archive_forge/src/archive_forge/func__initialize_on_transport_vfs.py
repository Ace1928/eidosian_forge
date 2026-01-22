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